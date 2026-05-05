# 🧠 KaStack RAG Chatbot

A fully local, offline-capable Retrieval-Augmented Generation (RAG) chatbot that
processes 11 000+ conversations, builds a persona profile, and lets you query
the dataset — **no paid APIs, no cloud inference required.**

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [How Topic Detection Works](#how-topic-detection-works)
3. [How Retrieval Works](#how-retrieval-works)
4. [How Persona Is Built](#how-persona-is-built)
5. [Setup](#setup)
6. [Running the App](#running-the-app)
7. [Hugging Face Spaces Deployment](#hugging-face-spaces-deployment)

---

## Architecture Overview

```
conversations.csv
      │
      ▼
rag_pipeline.py          ← flatten → embed → topic-detect → summarise → FAISS
      │
      ├── faiss_index/
      │       ├── topics.index   (topic-level embeddings)
      │       └── chunks.index   (100-message chunk embeddings)
      │
      └── checkpoints.json       (topic summaries + chunk texts)

persona_extractor.py     ← zero-shot classify + regex → persona.json

app.py                   ← Streamlit UI + dual FAISS retrieval + RoBERTa QA
```

---

## How Topic Detection Works

`rag_pipeline.py` uses a **cosine-similarity sliding window** approach:

1. Every message is embedded with `sentence-transformers/all-MiniLM-L6-v2`.
2. A sliding window of **N consecutive messages** computes pairwise cosine
   similarity between the embedding of the current window and the previous one.
3. When similarity drops **below a configurable threshold** (default `0.45`),
   a topic boundary is detected.
4. Each detected topic segment is then passed to a
   **BART-large-CNN summariser** (`facebook/bart-large-cnn`) to produce a
   one-sentence summary that becomes the topic node.

This means no LLM API calls — summarisation runs entirely on CPU/GPU locally.

---

## How Retrieval Works

A **dual FAISS index** is built at pipeline time:

| Index | Contents | Purpose |
|-------|----------|---------|
| `topics.index` | Embeddings of BART-generated topic summaries | Broad semantic matching |
| `chunks.index` | Embeddings of fixed 100-message windows | Precise excerpt retrieval |

At query time (`app.py`):

1. The user query is embedded with the same `all-MiniLM-L6-v2` model.
2. **Top-3 results** are retrieved from *both* FAISS indexes independently.
3. The six retrieved texts (3 topic summaries + 3 raw chunks) are concatenated
   with relevant persona signals into a single context string.
4. `deepset/roberta-base-squad2` (a fine-tuned extractive QA model) reads the
   context and extracts the most likely answer span.
5. The raw source chunks are displayed below the answer for transparency.

---

## How Persona Is Built

`persona_extractor.py` uses two complementary approaches:

### Zero-Shot Classification (facebook/bart-large-mnli)

- Conversations are processed in **batches of 100**.
- Each batch is classified into 6 candidate labels:
  `habits · hobbies · relationships · work · personality · communication style`
- Only predictions with **confidence ≥ 60%** are retained.
- Representative snippets are aggregated per label (up to 20 unique snippets).

### Rule-Based Regex Patterns

| Signal | Method |
|--------|--------|
| Emoji usage | `emoji.emoji_count()` across all conversations |
| Avg message length | Word-count per `User N:` turn |
| Sleep/time words | Regex: `morning, night, late, awake, midnight, …` |
| Food habits | Regex: `cook, bake, restaurant, pizza, chicken, …` |
| Job titles | Regex: `I'm a/an <job>` with 40+ occupation patterns |
| Pet names | Regex: `my dog/cat … is/named <Name>` |
| Family mentions | Regex: `my mom/dad/sister/brother/kids/…` |

> **Only signals explicitly present in the text are included — no guessing.**

The aggregated result is saved as `persona.json`:

```json
{
  "habits": [],
  "personal_facts": [],
  "personality_traits": [],
  "communication_style": {
    "avg_message_length": "",
    "tone": "",
    "emoji_usage": ""
  },
  "hobbies_sample": [],
  "relationships_sample": [],
  "work_sample": []
}
```

---

## Setup

### 1. Clone / enter the project directory

```bash
cd kastack-rag-chatbot
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key (optional — only needed if using Groq features)

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_actual_key_here
```

> **The core pipeline (indexing, persona, and the Streamlit app) works entirely without a GROQ key** — it uses only free local models. The `.env` file is for optional future Groq-powered features.

### 5. Build the pipeline

```bash
# Step 1: Build FAISS indexes + checkpoints (takes ~10–30 min on CPU)
python rag_pipeline.py

# Step 2: Build persona.json (takes ~15–60 min depending on hardware)
python persona_extractor.py
```

---

## Running the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Quick Questions available in the UI:**
- *"What kind of person is this user?"*
- *"What are their habits?"*
- *"How do they talk?"*

---

## Hugging Face Spaces Deployment

### Setting GROQ_API_KEY as a Space Secret

When deploying to [Hugging Face Spaces](https://huggingface.co/spaces), **never
hardcode API keys in your files.** Instead, inject them as Secrets:

1. Go to your Space → **Settings** tab → **Repository secrets** section.
2. Click **"New secret"**.
3. Set **Name** = `GROQ_API_KEY` and **Value** = your actual key.
4. Click **Save**.

At runtime, HF Spaces automatically injects the secret as an environment
variable. Your code retrieves it with:

```python
from dotenv import load_dotenv
import os
load_dotenv()                          # loads .env locally
api_key = os.getenv("GROQ_API_KEY")   # reads secret on HF Spaces
```

No hardcoding, no `.env` committed to Git. ✅

### Notes for HF Spaces

- Add `faiss-cpu` instead of `faiss-gpu` in `requirements.txt` (Spaces uses CPU).
- Upload pre-built `faiss_index/`, `checkpoints.json`, and `persona.json` as
  dataset files or use HF Datasets caching — don't re-build on every cold start.
- Set `SPACE_SDK=streamlit` and entry point to `app.py`.

---

## No API Keys or Paid Services Required

| Component | Model | Source |
|-----------|-------|--------|
| Embeddings | `all-MiniLM-L6-v2` | Hugging Face (free) |
| Topic summarisation | `facebook/bart-large-cnn` | Hugging Face (free) |
| Persona classification | `facebook/bart-large-mnli` | Hugging Face (free) |
| Question answering | `deepset/roberta-base-squad2` | Hugging Face (free) |
| Vector search | FAISS (CPU) | Meta / open-source |
| UI | Streamlit | Open-source |

All models are downloaded once and cached locally by `transformers` / `sentence-transformers`.
