"""
persona_extractor.py
────────────────────
Reads all conversations from conversations.csv and builds a persona.json
using:
  • facebook/bart-large-mnli  zero-shot classification (habits, hobbies, etc.)
  • Regex / rule-based pattern extraction (emojis, food words, sleep words, etc.)

No paid APIs — 100% local / free.
"""

from dotenv import load_dotenv
import os
load_dotenv()
# GROQ_API_KEY not used here (persona extraction is fully local)
# api_key = os.getenv("GROQ_API_KEY")

import re
import json
import csv
import emoji
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import pipeline

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH   = "conversations.csv"
OUTPUT     = "persona.json"
BATCH_SIZE = 100
ZS_MODEL   = "facebook/bart-large-mnli"

ZS_LABELS = ["habits", "hobbies", "relationships", "work",
             "personality", "communication style"]

# ── Regex Dictionaries ──────────────────────────────────────────────────────────
FOOD_WORDS = re.compile(
    r"\b(eat|food|cook|bake|recipe|restaurant|meal|breakfast|lunch|dinner|snack|"
    r"pizza|burger|sushi|pasta|taco|coffee|tea|drink|hungry|chef|bakery|dessert|"
    r"chicken|steak|salad|soup|sandwich|cookie|cake|bread)\b",
    re.IGNORECASE,
)

SLEEP_WORDS = re.compile(
    r"\b(morning|night|late|sleep|awake|midnight|early|insomnia|tired|bed|nap|"
    r"wake up|stayed up|up late|up early|good night|good morning)\b",
    re.IGNORECASE,
)

JOB_PATTERNS = re.compile(
    r"\b(I(?:'m| am) (?:a |an )?(?:software engineer|nurse|teacher|doctor|chef|"
    r"student|programmer|designer|manager|writer|artist|engineer|lawyer|dentist|"
    r"firefighter|police officer|EMT|musician|professor|trainer|barista|actor|"
    r"actress|photographer|accountant|pilot|librarian|coach|therapist|"
    r"veterinarian|scientist|researcher|developer|architect|director|"
    r"consultant|analyst|intern|freelancer|entrepreneur|owner|founder|"
    r"officer|agent|ranger|muralist|juggler|baker|decorator|model|blogger|tutor))\b",
    re.IGNORECASE,
)

PET_PATTERNS = re.compile(
    r"\b(?:my (?:dog|cat|pet|horse|bird|rabbit|hamster|fish|parrot|turtle|"
    r"goldfish|puppy|kitten|parakeet)(?:'s name)? (?:is |named )?(\w+)|"
    r"named (?:my (?:dog|cat|pet|horse|bird)) (\w+))\b",
    re.IGNORECASE,
)

FAMILY_WORDS = re.compile(
    r"\b(my (?:mom|dad|mother|father|sister|brother|son|daughter|wife|husband|"
    r"partner|grandma|grandpa|aunt|uncle|cousin|family|kids|children|baby|"
    r"spouse|boyfriend|girlfriend|fiancé|fiancée))\b",
    re.IGNORECASE,
)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def load_conversations(path: str) -> list[str]:
    """Read conversations.csv — each row is a full multi-turn conversation."""
    convs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                text = row[0].strip()
                if text:
                    convs.append(text)
    print(f"[✓] Loaded {len(convs)} conversations from {path}")
    return convs


def count_emojis(text: str) -> int:
    return emoji.emoji_count(text)


def avg_message_length(conversations: list[str]) -> float:
    """Average word count per User turn across all conversations."""
    lengths = []
    for conv in conversations:
        # Messages are lines starting with "User N:"
        for line in conv.splitlines():
            line = line.strip()
            if re.match(r"^User \d+:", line):
                msg = re.sub(r"^User \d+:\s*", "", line)
                lengths.append(len(msg.split()))
    return round(float(np.mean(lengths)) if lengths else 0.0, 2)


def extract_jobs(conversations: list[str]) -> list[str]:
    jobs = set()
    for conv in conversations:
        for match in JOB_PATTERNS.finditer(conv):
            raw = match.group(0)
            # strip leading "I'm a / I am an"
            cleaned = re.sub(r"^I(?:'m| am) (?:a |an )?", "", raw, flags=re.IGNORECASE).strip()
            jobs.add(cleaned.lower().capitalize())
    return sorted(jobs)


def extract_pet_names(conversations: list[str]) -> list[str]:
    pet_names = set()
    for conv in conversations:
        for match in PET_PATTERNS.finditer(conv):
            name = match.group(1) or match.group(2)
            if name:
                pet_names.add(name.capitalize())
    return sorted(pet_names)


def extract_family_mentions(conversations: list[str]) -> list[str]:
    mentions = set()
    for conv in conversations:
        for match in FAMILY_WORDS.finditer(conv):
            mentions.add(match.group(0).lower())
    return sorted(mentions)


def detect_food_interest(conversations: list[str]) -> bool:
    total_hits = sum(len(FOOD_WORDS.findall(c)) for c in conversations)
    # >1 hit per 10 conversations → food is a genuine interest
    return total_hits > len(conversations) / 10


def detect_sleep_patterns(conversations: list[str]) -> list[str]:
    hits = defaultdict(int)
    for conv in conversations:
        for word in SLEEP_WORDS.findall(conv):
            hits[word.lower()] += 1
    # Return words mentioned more than twice
    return sorted(k for k, v in hits.items() if v > 2)


def classify_batch(texts: list[str], classifier) -> dict[str, list[str]]:
    """
    Run zero-shot classification on a batch.
    Returns label → list of representative snippets.
    """
    results: dict[str, list[str]] = defaultdict(list)

    # Truncate each text to ~300 chars so the model doesn't choke
    truncated = [t[:300] for t in texts]
    preds = classifier(truncated, candidate_labels=ZS_LABELS, multi_label=True)

    for text, pred in zip(texts, preds):
        for label, score in zip(pred["labels"], pred["scores"]):
            if score >= 0.60:          # confident classification only
                snippet = text[:120].replace("\n", " ").strip()
                results[label].append(snippet)

    return results


# ── Main ────────────────────────────────────────────────────────────────────────

def build_persona():
    conversations = load_conversations(CSV_PATH)

    print("[→] Loading facebook/bart-large-mnli (zero-shot classifier)…")
    classifier = pipeline(
        "zero-shot-classification",
        model=ZS_MODEL,
        device=-1,          # CPU; set to 0 if you have a GPU
    )

    # Accumulate zero-shot signals
    zs_signals: dict[str, list[str]] = defaultdict(list)

    batches = [conversations[i: i + BATCH_SIZE]
               for i in range(0, len(conversations), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Classifying batches"):
        batch_results = classify_batch(batch, classifier)
        for label, snippets in batch_results.items():
            zs_signals[label].extend(snippets)

    # Deduplicate snippets — keep up to 20 per category
    for label in zs_signals:
        seen, unique = set(), []
        for s in zs_signals[label]:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        zs_signals[label] = unique[:20]

    # Rule-based extraction
    print("[→] Running rule-based extractors…")
    total_emojis  = sum(count_emojis(c) for c in conversations)
    avg_len       = avg_message_length(conversations)
    jobs          = extract_jobs(conversations)
    pet_names     = extract_pet_names(conversations)
    family_refs   = extract_family_mentions(conversations)
    food_interest = detect_food_interest(conversations)
    sleep_signals = detect_sleep_patterns(conversations)

    # Infer tone heuristically
    emoji_ratio = total_emojis / max(len(conversations), 1)
    if emoji_ratio > 2:
        tone = "expressive and emoji-heavy"
    elif avg_len < 12:
        tone = "concise and casual"
    elif avg_len > 30:
        tone = "verbose and detailed"
    else:
        tone = "conversational and balanced"

    emoji_usage = (
        f"{total_emojis} total emojis across {len(conversations)} conversations "
        f"(~{round(emoji_ratio, 2)} per conversation)"
    )

    # Habits — combine ZS + rule-based
    habits = list(zs_signals.get("habits", []))
    if food_interest:
        habits.append("Shows consistent interest in food, cooking, and restaurants")
    if sleep_signals:
        habits.append(
            "Sleep/time-related mentions: "
            + ", ".join(sleep_signals[:10])
        )

    # Personal facts — combine jobs, pets, family
    personal_facts = []
    if jobs:
        personal_facts.append("Mentioned job roles: " + ", ".join(jobs[:15]))
    if pet_names:
        personal_facts.append("Pet names found: " + ", ".join(pet_names[:10]))
    if family_refs:
        personal_facts.append(
            "Family references: " + ", ".join(sorted(set(family_refs))[:15])
        )

    # Personality — from ZS
    personality_traits = list(zs_signals.get("personality", []))[:15]

    # Assemble persona
    persona = {
        "habits":           habits[:20],
        "personal_facts":   personal_facts,
        "personality_traits": personality_traits,
        "communication_style": {
            "avg_message_length": f"{avg_len} words per message",
            "tone":               tone,
            "emoji_usage":        emoji_usage,
        },
        "hobbies_sample":       list(zs_signals.get("hobbies", []))[:15],
        "relationships_sample": list(zs_signals.get("relationships", []))[:10],
        "work_sample":          list(zs_signals.get("work", []))[:10],
        "_meta": {
            "total_conversations": len(conversations),
            "batch_size":          BATCH_SIZE,
            "zs_model":            ZS_MODEL,
            "extraction_note":     "Only explicitly observed signals are included.",
        },
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)

    print(f"[✓] Persona saved to {OUTPUT}")
    return persona


if __name__ == "__main__":
    build_persona()
