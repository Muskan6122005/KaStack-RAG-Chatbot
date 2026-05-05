"""
rag_pipeline.py
───────────────
Reads conversations.csv → embeds messages → detects topic shifts via cosine
similarity sliding window (threshold=0.35, window=5) → summarises each topic
segment with facebook/bart-large-cnn → extracts keywords with sklearn TF-IDF →
generates 100-message chunk summaries → builds dual FAISS index → saves
checkpoints.json and faiss_index/.

Run once:  python rag_pipeline.py
No paid APIs — 100% local / free.
"""

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")   # available but not used here

import csv
import json
import re
import torch
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging
logging.set_verbosity_error()
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Config ──────────────────────────────────────────────────────────────────────
CSV_PATH         = "conversations.csv"
CHECKPOINT_FILE  = "checkpoints.json"
FAISS_DIR        = "faiss_index"
EMBED_MODEL      = "all-MiniLM-L6-v2"
SUMMARISER_MODEL = "facebook/bart-large-cnn"

# Topic-detection (sliding window cosine similarity)
WINDOW_SIZE   = 5     # messages each side of the boundary
SIM_THRESHOLD = 0.35  # drop below this → new topic
MIN_TOPIC_SIZE = 10   # don't split until at least this many messages

# 100-message checkpoints
CHUNK_SIZE = 100

# FAISS retrieval counts (plan spec)
TOP_TOPICS = 5
TOP_CHUNKS = 10

os.makedirs(FAISS_DIR, exist_ok=True)


# ── 1. PARSING ──────────────────────────────────────────────────────────────────

def load_messages(path: str) -> list[dict]:
    """
    Flatten all CSV rows into a chronological list of messages tagged with
    (row_index, speaker, text).  Each CSV row is one multi-turn conversation.
    """
    messages = []
    row_index = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            conv_text = row[0]
            row_index += 1
            for line in conv_text.splitlines():
                line = line.strip()
                m = re.match(r"^(User \d+):\s*(.*)", line)
                if m and m.group(2).strip():
                    messages.append({
                        "row_index": row_index,
                        "speaker":   m.group(1),
                        "text":      m.group(2).strip(),
                    })
    print(f"[✓] Parsed {len(messages)} messages from {row_index} conversations")
    return messages


# ── 2. EMBEDDINGS ───────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], model: SentenceTransformer,
                batch_size: int = 256) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")


# ── 3. TOPIC DETECTION (sliding window cosine similarity) ───────────────────────

def detect_topic_boundaries(embeddings: np.ndarray) -> list[tuple[int, int]]:
    """
    Sliding window of WINDOW_SIZE messages.  When cosine similarity between
    the left and right window drops below SIM_THRESHOLD, mark a topic boundary.
    Returns list of (start_idx, end_idx) segment pairs.
    """
    n = len(embeddings)
    boundaries = [0]
    for i in range(WINDOW_SIZE, n - WINDOW_SIZE):
        left  = embeddings[i - WINDOW_SIZE: i].mean(axis=0)
        right = embeddings[i: i + WINDOW_SIZE].mean(axis=0)
        sim = float(
            np.dot(left, right) /
            (np.linalg.norm(left) * np.linalg.norm(right) + 1e-10)
        )
        if sim < SIM_THRESHOLD and (i - boundaries[-1]) >= MIN_TOPIC_SIZE:
            boundaries.append(i)
    boundaries.append(n)
    segments = [(boundaries[k], boundaries[k + 1])
                for k in range(len(boundaries) - 1)]
    print(f"[✓] Detected {len(segments)} topic segments (threshold={SIM_THRESHOLD})")
    return segments


# ── Summariser (direct AutoModel — 'summarization' pipeline removed in newer HF)

def load_summariser():
    print(f"[→] Loading summariser: {SUMMARISER_MODEL} …")
    tok = AutoTokenizer.from_pretrained(SUMMARISER_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(SUMMARISER_MODEL)
    mdl.eval()
    return tok, mdl


def summarise(texts: list[str], tok, mdl, max_input_tokens: int = 1024) -> str:
    combined = " ".join(texts)
    try:
        inputs = tok(
            combined,
            max_length=max_input_tokens,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=2,
                early_stopping=True,
            )
        return tok.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return combined[:150]


# ── TF-IDF keyword extraction ───────────────────────────────────────────────────

def extract_keywords(texts: list[str], n: int = 5) -> list[str]:
    """Return top-n TF-IDF keywords from a list of text strings."""
    corpus = " ".join(texts)
    if len(corpus.split()) < 5:
        return []
    try:
        vec = TfidfVectorizer(max_features=n, stop_words="english")
        vec.fit([corpus])
        return list(vec.get_feature_names_out())
    except Exception:
        return []


# ── 3. TOPIC CHECKPOINTS ────────────────────────────────────────────────────────

def build_topic_checkpoints(messages: list[dict], embeddings: np.ndarray,
                             tok, mdl) -> tuple[list[dict], np.ndarray]:
    """
    For each topic segment: generate summary + TF-IDF keywords.
    Stores: {topic_id, start_msg, end_msg, summary, keywords}
    """
    segments = detect_topic_boundaries(embeddings)
    topics   = []
    topic_embs = []

    print(f"[→] Summarising {len(segments)} topic segments …")
    for start, end in tqdm(segments, desc="Topics"):
        seg_texts = [messages[i]["text"] for i in range(start, end)]
        summary   = summarise(seg_texts, tok, mdl)
        keywords  = extract_keywords(seg_texts, n=5)
        seg_emb   = embeddings[start:end].mean(axis=0)

        topics.append({
            "topic_id":  len(topics),
            "start_msg": start,
            "end_msg":   end,
            "summary":   summary,
            "keywords":  keywords,
        })
        topic_embs.append(seg_emb)

    # Re-normalise averaged embeddings
    topic_matrix = np.array(topic_embs, dtype="float32")
    norms = np.linalg.norm(topic_matrix, axis=1, keepdims=True)
    topic_matrix = topic_matrix / (norms + 1e-10)

    print(f"[✓] {len(topics)} topic checkpoints built")
    return topics, topic_matrix


# ── 4. 100-MESSAGE CHECKPOINTS ──────────────────────────────────────────────────

def build_chunk_checkpoints(messages: list[dict], tok, mdl) -> list[dict]:
    """
    Every 100 messages, generate a BART summary.
    Stores: {checkpoint_id, msg_range, summary}
    """
    chunks = []
    total  = len(messages)
    print(f"[→] Building 100-message checkpoints …")
    for i in tqdm(range(0, total, CHUNK_SIZE), desc="Chunks"):
        batch   = messages[i: i + CHUNK_SIZE]
        texts   = [m["text"] for m in batch]
        summary = summarise(texts, tok, mdl)
        chunks.append({
            "checkpoint_id": len(chunks),
            "msg_range":     [i, i + len(batch)],
            "summary":       summary,
        })
    print(f"[✓] {len(chunks)} chunk checkpoints built")
    return chunks


# ── 5. VECTOR INDEX ─────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray, name: str) -> faiss.Index:
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product = cosine on normalised vecs
    index.add(embeddings)
    path  = os.path.join(FAISS_DIR, f"{name}.index")
    faiss.write_index(index, path)
    print(f"[✓] Saved {path}  ({len(embeddings)} vectors, dim={dim})")
    return index


# ── 6. QUERY FUNCTION ───────────────────────────────────────────────────────────

def query(q: str, embedder: SentenceTransformer,
          topic_index: faiss.Index, chunk_index: faiss.Index,
          topics: list[dict], chunks: list[dict]) -> str:
    """
    Embed query → retrieve top-5 topic summaries + top-10 chunk summaries
    → return combined context string.
    """
    q_emb = embedder.encode([q], normalize_embeddings=True).astype("float32")

    _, t_ids = topic_index.search(q_emb, TOP_TOPICS)
    _, c_ids = chunk_index.search(q_emb, TOP_CHUNKS)

    parts = []
    for idx in t_ids[0]:
        if idx < len(topics):
            t = topics[idx]
            parts.append(f"[Topic summary] {t['summary']}  keywords: {', '.join(t['keywords'])}")
    for idx in c_ids[0]:
        if idx < len(chunks):
            c = chunks[idx]
            parts.append(f"[Chunk {c['checkpoint_id']} msgs {c['msg_range']}] {c['summary']}")

    return "\n\n".join(parts)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    # 1. Parse
    messages = load_messages(CSV_PATH)

    # 2. Embed all messages
    print(f"[→] Loading embedder: {EMBED_MODEL} …")
    embedder   = SentenceTransformer(EMBED_MODEL)
    msg_texts  = [m["text"] for m in messages]
    print(f"[→] Embedding {len(msg_texts)} messages …")
    all_embs   = embed_texts(msg_texts, embedder)

    # Load summariser once
    tok, mdl = load_summariser()

    # 3. Topic checkpoints
    topics, topic_matrix = build_topic_checkpoints(messages, all_embs, tok, mdl)

    # 4. 100-message checkpoints
    chunks = build_chunk_checkpoints(messages, tok, mdl)

    # 5. Embed chunk summaries for FAISS
    print("[→] Embedding chunk summaries …")
    chunk_summary_texts = [c["summary"] for c in chunks]
    chunk_embs = embed_texts(chunk_summary_texts, embedder, batch_size=64)

    # Build FAISS indexes
    topic_index = build_faiss_index(topic_matrix, "topics")
    chunk_index = build_faiss_index(chunk_embs,   "chunks")

    # Save checkpoints.json
    checkpoint = {
        "topics": topics,
        "chunks": chunks,
        "meta": {
            "total_messages":    len(messages),
            "total_topics":      len(topics),
            "total_chunks":      len(chunks),
            "embed_model":       EMBED_MODEL,
            "summariser_model":  SUMMARISER_MODEL,
            "sim_threshold":     SIM_THRESHOLD,
            "window_size":       WINDOW_SIZE,
            "chunk_size":        CHUNK_SIZE,
        },
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Done!")
    print(f"    faiss_index/topics.index  ({len(topics)} topic vectors)")
    print(f"    faiss_index/chunks.index  ({len(chunks)} chunk vectors)")
    print(f"    checkpoints.json")

    # Quick sanity check
    print("\n[→] Sanity-checking query function …")
    result = query("What are this person's hobbies?", embedder,
                   topic_index, chunk_index, topics, chunks)
    print(result[:400])


if __name__ == "__main__":
    main()
