"""
Volta Electrical Services Ltd RAG demo — agent.

Two tools, one LangGraph ReAct agent (langgraph.prebuilt.create_react_agent):
  - search_knowledge_base(query)  -> semantic search over the ingested PDF (Qdrant)
  - lookup_job(job_reference)     -> mock live job/booking status (jobs.json)

The model decides which tool(s) a question needs:
  - "What's included in a consumer unit upgrade?" -> search_knowledge_base only
  - "What's the status of my job VLT-1003?"       -> lookup_job only
  - "My job is VLT-1008 — what certificate did I get?" -> both tools

app.py imports answer() from here — nothing in this file is meant to be
run directly.
"""

import json
import os

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

QDRANT_URL     = os.environ["QDRANT_URL"].rstrip("/")
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION     = os.environ.get("QDRANT_COLLECTION", "volta_demo")
GROQ_API_KEY   = os.environ["GROQ_API_KEY"]
MAIN_MODEL     = os.environ.get("MAIN_MODEL", "openai/gpt-oss-120b")

# bge-small-en-v1.5 (33M params, ~130MB, 384-dim).
# Forced to CPU — letting sentence-transformers auto-detect CUDA on a
# shared / virtual GPU produced NaN vectors in earlier deployments.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
JOBS_PATH        = os.path.join(os.path.dirname(__file__), "jobs.json")
TOP_K            = 4

HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}

# Load once at import time — avoids repeated 1-2 s model loads per request.
_embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

with open(JOBS_PATH) as f:
    _JOBS = {j["job_reference"]: j for j in json.load(f)}


@tool
def search_knowledge_base(query: str) -> str:
    """Search Volta Electrical Services Ltd's knowledge base — services,
    pricing, certifications, EICR, EV chargers, solar PV, consumer unit
    upgrades, PAT testing, landlord compliance, Part P regulations, booking
    policy, guarantees, and FAQ.

    Use this for ANY question about what Volta does, how much it costs,
    how long something takes, what a certificate means, or what the rules
    are. Always call this before answering a policy, pricing, or services
    question — never answer from memory."""

    # bge-small-en-v1.5 requires this prefix on QUERIES only.
    # Documents ingested by ingest.py are NOT prefixed.
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vector   = _embed_model.encode(prefixed, normalize_embeddings=True).tolist()

    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        headers=HEADERS,
        json={"vector": vector, "limit": TOP_K, "with_payload": True},
        timeout=30,
    )
    resp.raise_for_status()

    hits = resp.json()["result"]
    if not hits:
        return "No matching content found in Volta's knowledge base."

    blocks = [
        f"[page {h['payload']['page']}] {h['payload']['text']}"
        for h in hits
    ]
    return "\n\n".join(blocks)


@tool
def lookup_job(job_reference: str) -> str:
    """Look up the live status of a Volta Electrical job by its reference
    number, e.g. "VLT-1003".  Use this whenever a customer asks about the
    status, engineer, scheduled date, certificate, or invoice for a specific
    job.  Never guess a job's status from the knowledge base — only
    lookup_job contains real job data."""

    ref   = job_reference.strip().upper()
    # Tolerate "1003" as well as "VLT-1003"
    if not ref.startswith("VLT-"):
        ref = f"VLT-{ref}"

    job = _JOBS.get(ref)
    if not job:
        return (
            f"No job found with reference {ref}. "
            "Please ask the customer to double-check the reference in their "
            "booking confirmation email."
        )
    return json.dumps(job, indent=2)


SYSTEM_PROMPT = """You are the customer support assistant for Volta Electrical \
Services Ltd, a NICEIC-approved electrical contractor based in Croydon, \
South London (phone: 0208 555 0174, emergency 24/7: 07700 900 447).

Answer ONLY from what search_knowledge_base and lookup_job return — never \
invent a price, policy, certificate, or job status.

Rules:
- For any question about services, pricing, EICR, Part P, EV chargers, solar, \
  PAT testing, booking, guarantee, or compliance: call search_knowledge_base first.
- For any question about a specific job (status, engineer, certificate, invoice): \
  call lookup_job with the job reference (VLT-XXXX format).
- If the question needs both (e.g. "My job is VLT-1002 — what do I have to do \
  about the C2 findings?"): call both tools before answering.
- Keep answers short and direct, in plain customer-support language.
- If nothing relevant is found, say so honestly instead of guessing.
- When quoting a price or regulation, mention which section of the knowledge \
  base it came from if the page number is available."""


_model = ChatGroq(model=MAIN_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
try:
    _graph = create_react_agent(_model, tools=[search_knowledge_base, lookup_job],
                                state_modifier=SYSTEM_PROMPT)
except TypeError:
    # Older langgraph uses 'prompt' instead of 'state_modifier'
    _graph = create_react_agent(_model, tools=[search_knowledge_base, lookup_job],
                                prompt=SYSTEM_PROMPT)


def answer(message: str, history: list) -> str:
    """history is a list of {"role": "user"/"assistant", "content": str} dicts."""
    messages = []
    for turn in history:
        role    = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=message))

    result = _graph.invoke({"messages": messages})

    for msg in reversed(result["messages"]):
        content = getattr(msg, "content", "")
        if isinstance(msg, AIMessage) and content:
            return content
    return "Sorry, I couldn't generate an answer — please try rephrasing."