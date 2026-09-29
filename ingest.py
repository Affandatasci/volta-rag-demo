"""
Volta Electrical Services Ltd RAG demo — one-time ingestion script.

Reads volta_electrical_faq.pdf, splits it into chunks, embeds each chunk
with bge-small-en-v1.5 (forced to CPU — avoids ZeroGPU virtual-CUDA
producing NaN vectors), and upserts everything into Qdrant Cloud over
the plain REST API.

Using bge-small (33M params, ~130MB) instead of mxbai-embed-large-v1
(335M params, ~1.3GB) to stay inside Streamlit Community Cloud's free-
tier RAM limit (~2.7 GB).  The same "prefix the query, not the documents"
convention applies.

Deliberately not using qdrant-client's search()/query_points(): search()
was removed in v1.16.0 and query_points() is rejected by some server
versions.  This script calls Qdrant's REST endpoint directly with
`requests` for both writing and reading.

Run once before app.py is used:
    python ingest.py

Requires QDRANT_URL, QDRANT_API_KEY (see .env.example).
Set QDRANT_COLLECTION to a name distinct from any other demo on the
same cluster (default: volta_demo).
"""

import os
import sys
import uuid

import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

PDF_PATH         = os.path.join(os.path.dirname(__file__), "volta_electrical_faq.pdf")
QDRANT_URL       = os.environ["QDRANT_URL"].rstrip("/")
QDRANT_API_KEY   = os.environ["QDRANT_API_KEY"]
COLLECTION       = os.environ.get("QDRANT_COLLECTION", "volta_demo")
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM        = 384
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 120

HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}


def extract_text(pdf_path):
    """Return [{"page": n, "text": "..."}] for every page with real text."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_pages(pages):
    """Split each page's text into overlapping chunks tagged with the source
    page number so answers can point back to the original document."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for p in pages:
        for piece in splitter.split_text(p["text"]):
            piece = piece.strip()
            if piece:
                chunks.append({"text": piece, "page": p["page"]})
    return chunks


def reset_collection():
    """Delete and recreate the collection — safe to re-run."""
    requests.delete(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers=HEADERS,
        timeout=30,
    )
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers=HEADERS,
        json={"vectors": {"size": EMBED_DIM, "distance": "Cosine"}},
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create collection: {resp.status_code} {resp.text}")
    print(f"Collection '{COLLECTION}' created fresh.")


def upsert_chunks(chunks, embeddings):
    points = []
    for chunk, vector in zip(chunks, embeddings):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector.tolist(),
            "payload": {
                "text":   chunk["text"],
                "page":   chunk["page"],
                "source": "Volta Electrical Services Ltd — Services, Pricing & FAQ",
            },
        })
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
        headers=HEADERS,
        json={"points": points},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to upsert points: {resp.status_code} {resp.text}")
    print(f"Upserted {len(points)} chunks.")


def main():
    if not os.path.exists(PDF_PATH):
        sys.exit(
            f"Can't find {PDF_PATH}\n"
            "Put volta_electrical_faq.pdf in the same directory as ingest.py first."
        )

    print("Reading PDF...")
    pages = extract_text(PDF_PATH)
    print(f"Extracted text from {len(pages)} pages.")

    print("Chunking...")
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading embedding model ({EMBED_MODEL_NAME}) on CPU...")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    print("Embedding chunks...")
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("Resetting Qdrant collection...")
    reset_collection()

    print("Upserting into Qdrant...")
    upsert_chunks(chunks, embeddings)

    print("Done — agent.py can now query the collection.")


if __name__ == "__main__":
    main()
