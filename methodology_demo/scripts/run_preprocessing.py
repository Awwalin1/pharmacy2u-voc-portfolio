"""
run_preprocessing.py

End-to-end pipeline runner. Loads the synthetic contacts, applies
channel-aware preprocessing, and writes the cleaned output alongside.

Demonstrates the pipeline runs end-to-end:
    raw synthetic contacts  →  channel-aware preprocessing  →  cleaned contacts

Run from project root:
    python methodology_demo/scripts/run_preprocessing.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Add project root to path so we can import the preprocessing module
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from methodology_demo.preprocessing.channel_aware import preprocess


# ---------------------------------------------------------------------------
# Paths — relative to project root
# ---------------------------------------------------------------------------
INPUT_PATH = PROJECT_ROOT / "methodology_demo" / "data" / "synthetic_contacts.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "methodology_demo" / "data" / "cleaned_contacts.jsonl"

DISCLAIMER = (
    "DEMO DATA — illustrative only. Cleaned via channel-aware preprocessing. "
    "Does not represent real customer interactions."
)


def load_records(path: Path):
    """Stream records from JSONL, skipping the header line."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_HEADER" in obj:
                continue  # skip disclaimer header
            records.append(obj)
    return records


def write_records(records, path: Path):
    """Write records to JSONL with a fresh disclaimer header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_HEADER": DISCLAIMER}) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarise(records, cleaned):
    """Print a short summary so the pipeline run is visible at the terminal."""
    channels = Counter(r["channel"] for r in records)
    intents = Counter(r["intent"] for r in records)

    print(f"\nLoaded {len(records)} records from {INPUT_PATH.name}")
    print(f"Wrote {len(cleaned)} cleaned records to {OUTPUT_PATH.name}\n")

    print("Channel distribution:")
    for k, v in channels.most_common():
        print(f"  {k:8s} {v:3d}")

    print("\nIntent distribution:")
    for k, v in intents.most_common():
        print(f"  {k:35s} {v:3d}")

    # Show one before/after example per channel — gives a visible sanity check
    print("\nBefore / after preview (one example per channel):\n")
    seen = set()
    for orig, clean in zip(records, cleaned):
        ch = orig["channel"]
        if ch in seen:
            continue
        seen.add(ch)
        before = orig["text"].replace("\n", " ⏎ ")
        after = clean["cleaned_text"]
        print(f"--- {ch} | {orig['intent']} ---")
        print(f"  before: {before[:140]}{'…' if len(before) > 140 else ''}")
        print(f"  after:  {after[:140]}{'…' if len(after) > 140 else ''}")
        print()
        if len(seen) == 4:
            break


def main():
    if not INPUT_PATH.exists():
        print(f"ERROR: input file not found at {INPUT_PATH}")
        print("Run methodology_demo/scripts/synthetic_contact_generator.py first.")
        sys.exit(1)

    records = load_records(INPUT_PATH)
    cleaned = [preprocess(r) for r in records]
    write_records(cleaned, OUTPUT_PATH)
    summarise(records, cleaned)


if __name__ == "__main__":
    main()