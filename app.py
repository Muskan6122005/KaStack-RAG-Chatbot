"""
app.py
──────
Streamlit RAG Chatbot — 100% local, no paid APIs.

Startup:  streamlit run app.py

Step 1:   python rag_pipeline.py      → builds faiss_index/ + checkpoints.json
Step 2:   python persona_extractor.py → builds persona.json
Step 3:   streamlit run app.py
"""

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")   # loaded but not used (all local)

import json
import torch
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KaStack RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
}
.answer-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 20px 24px;
    margin-top: 14px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    color: #f0f0f0;
    line-height: 1.7;
}
.source-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid #7c3aed;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 8px;
    color: #ccc;
    font-size: 0.85rem;
}
.pill {
    display: inline-block;
    background: rgba(124,58,237,0.25);
    border: 1px solid rgba(124,58,237,0.5);
    border-radius: 999px;
    padding: 3px 12px;
    margin: 3px 2px;
    font-size: 0.78rem;
    color: #ddd;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.45) !important;
}
h1, h2, h3 { color: #f0f0f0 !important; }
p, li, label { color: #ccc !important; }
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #f0f0f0 !important;
    padding: 12px 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading sentence-transformer …")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading QA model (deepset/roberta-base-squad2) …")
def load_qa_model():
    """Load deepset/roberta-base-squad2 via AutoModel (pipeline task removed in newer HF)."""
    tok = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    mdl = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    mdl.eval()
    return tok, mdl


@st.cache_data(show_spinner="Loading FAISS index & checkpoints …")
def load_index_and_checkpoints():
    required = [
        "faiss_index/topics.index",
        "faiss_index/chunks.index",
        "checkpoints.json",
    ]
    if any(not os.path.exists(r) for r in required):
        return None, None, None, None

    topic_index = faiss.read_index("faiss_index/topics.index")
    chunk_index = faiss.read_index("faiss_index/chunks.index")

    with open("checkpoints.json", encoding="utf-8") as f:
        cp = json.load(f)

    topics = cp.get("topics", [])   # {topic_id, start_msg, end_msg, summary, keywords}
    chunks = cp.get("chunks", [])   # {checkpoint_id, msg_range, summary}

    return topic_index, chunk_index, topics, chunks


@st.cache_data(show_spinner="Loading persona …")
def load_persona():
    if not os.path.exists("persona.json"):
        return {}
    with open("persona.json", encoding="utf-8") as f:
        return json.load(f)


# ── FAISS retrieval (top-5 topics + top-10 chunks per plan) ────────────────────

def retrieve(query_str: str, embedder,
             topic_index, chunk_index, topics, chunks):
    q_emb = embedder.encode([query_str], normalize_embeddings=True).astype("float32")

    _, t_ids = topic_index.search(q_emb, 5)
    _, c_ids = chunk_index.search(q_emb, 10)

    top_topics = [topics[i] for i in t_ids[0] if i < len(topics)]
    top_chunks = [chunks[i] for i in c_ids[0] if i < len(chunks)]

    return top_topics, top_chunks


def build_context(query_str: str, persona: dict,
                  top_topics: list, top_chunks: list) -> str:
    parts = []

    # Persona signals
    cs = persona.get("communication_style", {})
    if cs:
        parts.append(
            f"[Communication Style] Avg message length: {cs.get('avg_message_length','')}. "
            f"Tone: {cs.get('tone','')}. Emoji usage: {cs.get('emoji_usage','')}."
        )
    if persona.get("personality_traits"):
        parts.append("[Personality] " + " | ".join(persona["personality_traits"][:5]))
    if persona.get("habits"):
        parts.append("[Habits] " + " | ".join(persona["habits"][:5]))
    if persona.get("personal_facts"):
        parts.append("[Facts] " + " | ".join(persona["personal_facts"][:3]))

    # Top-5 topic summaries
    for t in top_topics:
        kw = ", ".join(t.get("keywords", []))
        parts.append(f"[Topic summary] {t['summary']}  (keywords: {kw})")

    # Top-10 chunk summaries
    for c in top_chunks:
        parts.append(f"[Chunk {c['checkpoint_id']} msgs {c['msg_range']}] {c['summary']}")

    return "\n\n".join(parts)


# ── QA inference (direct AutoModel) ───────────────────────────────────────────

def answer_question(question: str, context: str, tok, mdl) -> str:
    if not context.strip():
        return "Not enough context found in the data."

    # roberta-base-squad2 max = 512 tokens; truncate context to be safe
    ctx = context[:2000]
    try:
        inputs = tok(
            question, ctx,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = mdl(**inputs)

        start = torch.argmax(outputs.start_logits)
        end   = torch.argmax(outputs.end_logits) + 1

        answer = tok.convert_tokens_to_string(
            tok.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
        ).strip()

        if not answer or answer in ("[CLS]", "[SEP]", "<s>", "</s>"):
            answer = "I found relevant context but couldn't extract a precise answer — see sources below."

        return answer
    except Exception as e:
        return f"QA error: {e}"


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(persona: dict):
    st.sidebar.markdown("## 🧠 Persona Highlights")

    cs = persona.get("communication_style", {})
    if cs:
        st.sidebar.markdown("**Communication Style**")
        for k, v in cs.items():
            st.sidebar.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")

    if persona.get("habits"):
        st.sidebar.markdown("---\n**Detected Habits**")
        for h in persona["habits"][:6]:
            st.sidebar.markdown(f'<div class="pill">{h[:80]}</div>', unsafe_allow_html=True)

    if persona.get("personal_facts"):
        st.sidebar.markdown("---\n**Personal Facts**")
        for fact in persona["personal_facts"][:4]:
            st.sidebar.markdown(f"- {fact[:100]}")

    if persona.get("personality_traits"):
        st.sidebar.markdown("---\n**Personality Traits**")
        for t in persona["personality_traits"][:5]:
            st.sidebar.markdown(f'<div class="pill">{t[:80]}</div>', unsafe_allow_html=True)

    meta = persona.get("_meta", {})
    if meta:
        st.sidebar.markdown("---")
        st.sidebar.caption(
            f"📊 {meta.get('total_conversations','?')} conversations · "
            f"model: {meta.get('zs_model','N/A')}"
        )


# ── Main UI ────────────────────────────────────────────────────────────────────

def main():
    # Load models
    embedder          = load_embedder()
    qa_tok, qa_mdl    = load_qa_model()

    # Load data
    topic_index, chunk_index, topics, chunks = load_index_and_checkpoints()
    persona = load_persona()

    index_ready = topic_index is not None

    # Sidebar
    render_sidebar(persona)

    # Header
    st.markdown("""
    <h1 style='text-align:center;
               background:linear-gradient(90deg,#7c3aed,#4f46e5);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               font-size:2.4rem; margin-bottom:4px;'>
      🧠 KaStack RAG Chatbot
    </h1>
    <p style='text-align:center; color:#aaa; margin-bottom:28px;'>
      Ask anything about the conversation dataset — powered by local AI, no paid APIs.
    </p>
    """, unsafe_allow_html=True)

    if not index_ready:
        st.warning(
            "⚠️ **FAISS index not found.**  "
            "Run `python rag_pipeline.py` first to build `faiss_index/` and `checkpoints.json`.",
        )
        st.info("The Persona sidebar is still available if `persona.json` exists.")
        return

    # ── Example buttons ───────────────────────────────────────────────────────
    st.markdown("### 💡 Quick Questions")
    col1, col2, col3 = st.columns(3)
    example_query = None
    with col1:
        if st.button("What kind of person is this user?", key="btn_person"):
            example_query = "What kind of person is this user?"
    with col2:
        if st.button("What are their habits?", key="btn_habits"):
            example_query = "What are their habits?"
    with col3:
        if st.button("How do they talk?", key="btn_talk"):
            example_query = "How do they talk?"

    # ── Text input ────────────────────────────────────────────────────────────
    st.markdown("### 🔍 Ask a Question")
    user_input = st.text_input(
        "Question",
        value=example_query or "",
        placeholder="e.g. What hobbies does this person have?",
        label_visibility="collapsed",
        key="main_input",
    )
    ask_clicked = st.button("Ask →", key="ask_btn")

    query_str = (example_query or user_input).strip()

    if (ask_clicked or example_query) and query_str:
        with st.spinner("Retrieving context and generating answer …"):
            # Step 1 — FAISS retrieval: top-5 topic summaries + top-10 chunks
            top_topics, top_chunks = retrieve(
                query_str, embedder,
                topic_index, chunk_index, topics, chunks,
            )
            # Step 2 — Combine with persona
            context = build_context(query_str, persona, top_topics, top_chunks)
            # Step 3 — RoBERTa QA
            answer  = answer_question(query_str, context, qa_tok, qa_mdl)

        # Step 4 — Display answer + source chunks
        st.markdown("### 🤖 Answer")
        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

        all_sources = (
            [(t["summary"], "📌 Topic Summary",
              ", ".join(t.get("keywords", []))) for t in top_topics] +
            [(c["summary"], "💬 Chunk Summary",
              f"msgs {c['msg_range']}") for c in top_chunks]
        )
        if all_sources:
            st.markdown("#### 📚 Source Context")
            for text, label, meta_info in all_sources:
                if text.strip():
                    st.markdown(
                        f'<div class="source-card"><b>{label}</b> '
                        f'<span style="color:#888;font-size:0.8rem">{meta_info}</span>'
                        f'<br>{text[:400]}</div>',
                        unsafe_allow_html=True,
                    )

    elif ask_clicked and not query_str:
        st.warning("Please enter a question first.")


if __name__ == "__main__":
    main()
