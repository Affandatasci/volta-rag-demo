"""
Volta Electrical Services Ltd RAG demo — Streamlit UI.

Setup:
    1. pip install -r requirements.txt
    2. cp .env.example .env  and fill in all keys
    3. python ingest.py            (one-time: embeds the PDF into Qdrant)
    4. streamlit run app.py        (starts the chat UI)

Deploy on Streamlit Community Cloud (free, no card required):
    - Push this folder to a GitHub repo (public or private).
    - Connect at share.streamlit.io → New app → set main file to app.py.
    - Add GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION as
      root-level (not nested) keys in Advanced settings → Secrets.
    - Never commit real key values to the repo.
"""

import streamlit as st
from agent import answer

TITLE       = "Volta Electrical Services Ltd — Customer Assistant"
DESCRIPTION = (
    "Ask about our services, pricing, certifications, EICR, EV chargers, "
    "solar PV, or Part P compliance — or check the status of a specific job "
    "(try VLT-1001 through VLT-1010)."
)

# ── Palette: dark green banner + light green page ─────────────────────────────
DARK_GREEN       = "#1B4332"   # banner background
DARK_GREEN_MID   = "#2D6A4F"   # banner gradient midpoint
YELLOW           = "#F5C518"   # title accent
WHITE            = "#FFFFFF"

PAGE_BG          = "#E8F5E9"   # light green page background
BUBBLE_USER_BG   = "#C8E6C9"   # user bubble — medium green
BUBBLE_USER_BDR  = "#81C784"   # user bubble border
BUBBLE_ASST_BG   = "#FFFFFF"   # assistant bubble — white
BUBBLE_ASST_BDR  = "#A5D6A7"   # assistant bubble border
TEXT_DARK        = "#1B2B1E"   # main text colour on light green
BTN_BG           = DARK_GREEN
BTN_TEXT         = YELLOW

st.set_page_config(page_title=TITLE, page_icon="⚡", layout="centered")

CUSTOM_CSS = f"""
<style>
/* ── Page background ── */
html, body, [class*="css"], .stApp, .main, section.main > div {{
    background-color: {PAGE_BG} !important;
}}

/* ── Base font ── */
html, body, [class*="css"] {{
    font-size: 21px !important;
    color: {TEXT_DARK} !important;
}}

/* ── Banner ── */
.vl-banner {{
    background: linear-gradient(90deg, {DARK_GREEN}, {DARK_GREEN_MID}, {DARK_GREEN});
    border-radius: 14px;
    padding: 20px 26px 18px 26px;
    margin-bottom: 22px;
    border-left: 6px solid {YELLOW};
}}
.vl-banner h1 {{
    font-size: 38px !important;
    font-weight: 800 !important;
    color: {YELLOW} !important;
    margin: 0 0 8px 0 !important;
}}
.vl-banner p {{
    font-size: 20px !important;
    color: {WHITE} !important;
    font-weight: 400 !important;
    opacity: 0.93;
    margin: 0 !important;
}}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {{
    font-size: 21px !important;
    border-radius: 12px !important;
}}
[data-testid="stChatMessage"] p {{
    font-size: 21px !important;
    line-height: 1.6 !important;
}}

/* User bubble — medium green */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
    background: {BUBBLE_USER_BG} !important;
    border: 1px solid {BUBBLE_USER_BDR} !important;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] p {{
    color: {TEXT_DARK} !important;
}}

/* Assistant bubble — white */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
    background: {BUBBLE_ASST_BG} !important;
    border: 1px solid {BUBBLE_ASST_BDR} !important;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] p {{
    color: {TEXT_DARK} !important;
}}

/* ── Input box ── */
[data-testid="stChatInput"] textarea {{
    font-size: 21px !important;
    background: {WHITE} !important;
}}

/* ── Clear button ── */
.stButton > button {{
    background: {BTN_BG} !important;
    color: {BTN_TEXT} !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border-radius: 8px !important;
    border: none !important;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    f'<div class="vl-banner"><h1>⚡ {TITLE}</h1><p>{DESCRIPTION}</p></div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input(
    "Ask about services, pricing, EICR, EV chargers, Part P, or a job reference..."
)
if prompt:
    history_snapshot = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Looking that up..."):
            reply = answer(prompt, history_snapshot)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})