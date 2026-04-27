"""
run_sentiment.py

End-to-end sentiment scoring on the synthetic corpus.

  1. Loads cleaned synthetic contacts.
  2. Scores each contact with the RoBERTa sentiment model.
  3. Reports overall sentiment distribution.
  4. Cross-tabulates sentiment by intent and by channel.
  5. Surfaces the highest-confidence negative contacts as 'high concern'.

Run from project root (with .venv activated):
    python methodology_demo/scripts/run_sentiment.py
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from methodology_demo.sentiment.sentiment_scorer import SentimentScorer


CLEANED_PATH = PROJECT_ROOT / "methodology_demo" / "data" / "cleaned_contacts.jsonl"
SENTIMENT_OUTPUT_PATH = (
    PROJECT_ROOT / "methodology_demo" / "data" / "scored_contacts.jsonl"
)

DISCLAIMER = (
    "DEMO DATA - illustrative only. Sentiment scored via RoBERTa "
    "(cardiffnlp/twitter-roberta-base-sentiment-latest). Does not "
    "represent real customer interactions."
)


def load_records(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_HEADER" in obj:
                continue
            records.append(obj)
    return records


def write_scored(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_HEADER": DISCLAIMER}) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_overall_distribution(records):
    counts = Counter(r["sentiment_label"] for r in records)
    total = len(records)
    print("\n=== Overall sentiment distribution ===")
    for label in ["negative", "neutral", "positive"]:
        n = counts.get(label, 0)
        pct = 100.0 * n / total if total else 0.0
        print(f"  {label:<10s} {n:>3d}  ({pct:.1f}%)")


def print_crosstab(records, group_key, title):
    table = defaultdict(lambda: Counter())
    for r in records:
        table[r[group_key]][r["sentiment_label"]] += 1

    print(f"\n=== Sentiment by {title} ===")
    print(f"{title:<35s} {'neg':>5s} {'neu':>5s} {'pos':>5s} {'total':>7s}")
    print("-" * 60)
    for group, counts in sorted(table.items(), key=lambda x: -sum(x[1].values())):
        n_neg = counts.get("negative", 0)
        n_neu = counts.get("neutral", 0)
        n_pos = counts.get("positive", 0)
        total = n_neg + n_neu + n_pos
        print(f"{group:<35s} {n_neg:>5d} {n_neu:>5d} {n_pos:>5d} {total:>7d}")


def print_negative_examples(records, n_examples=3):
    negatives = [r for r in records if r["sentiment_label"] == "negative"]
    negatives.sort(key=lambda r: -r["sentiment_confidence"])

    print(f"\n=== Top {min(n_examples, len(negatives))} highest-confidence negative contacts ===")
    for r in negatives[:n_examples]:
        text_preview = r["cleaned_text"][:110]
        print(f"\n  intent: {r['intent']}  channel: {r['channel']}  "
              f"confidence: {r['sentiment_confidence']:.3f}")
        print(f"  text:   {text_preview}{'...' if len(r['cleaned_text']) > 110 else ''}")


def main():
    if not CLEANED_PATH.exists():
        print(f"ERROR: cleaned file not found at {CLEANED_PATH}")
        print("Run methodology_demo/scripts/run_preprocessing.py first.")
        sys.exit(1)

    print("Loading cleaned contacts...")
    records = load_records(CLEANED_PATH)
    print(f"  {len(records)} records loaded")

    print("\nLoading sentiment model (first run downloads ~500MB)...")
    scorer = SentimentScorer()

    print("Scoring contacts...")
    texts = [r["cleaned_text"] for r in records]
    scores = scorer.score(texts, batch_size=16)

    for record, score in zip(records, scores):
        record["sentiment_label"] = score["label"]
        record["sentiment_confidence"] = score["confidence"]
        record["sentiment_scores"] = score["scores"]

    print(f"  {len(records)} contacts scored")

    write_scored(records, SENTIMENT_OUTPUT_PATH)
    print(f"  Written to {SENTIMENT_OUTPUT_PATH.name}")

    print_overall_distribution(records)
    print_crosstab(records, "intent", "intent")
    print_crosstab(records, "channel", "channel")
    print_negative_examples(records, n_examples=3)

    print("\nDone.")


if __name__ == "__main__":
    main()