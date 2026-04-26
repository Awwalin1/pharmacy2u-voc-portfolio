"""
run_topics.py

End-to-end topic modelling demonstration on the synthetic corpus.

Run from project root (with .venv activated):
    python methodology_demo/scripts/run_topics.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from methodology_demo.topics.bertopic_modeller import TopicModeller


CLEANED_PATH = PROJECT_ROOT / "methodology_demo" / "data" / "cleaned_contacts.jsonl"


def load_records(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_HEADER" in obj:
                continue
            records.append(obj)
    return records


def print_topic_summary(modeller):
    info = modeller.topic_info()
    top_words = modeller.top_words_per_topic(n_words=6)
    print(f"\nDiscovered {len(top_words)} topic clusters (excluding outliers).\n")
    print(f"{'Topic':<10} {'Count':<8} {'Top words':<60}")
    print("-" * 78)
    for _, row in info.iterrows():
        topic_id = int(row["Topic"])
        count = int(row["Count"])
        if topic_id == -1:
            label = "outliers"
            words = "no consistent cluster"
        else:
            label = f"Topic {topic_id}"
            words = ", ".join(top_words.get(topic_id, []))
        print(f"{label:<10} {count:<8} {words[:60]}")


def print_crosstab(modeller):
    crosstab = modeller.cross_tab_against_intents()
    all_intents = sorted({
        intent for intents in crosstab.values() for intent in intents
    })
    print("\n=== Discovered topics vs hand-coded intents ===")
    print("Each row is a discovered topic. Each column is a hand-coded intent.")
    print("A topic that aligns with the taxonomy will be dominated by one intent.\n")
    short_intents = [i.replace("_", " ").lower()[:14] for i in all_intents]
    header_cells = " ".join(f"{si:>14}" for si in short_intents)
    print(f"{'Topic':<10} {header_cells}")
    print("-" * (10 + len(header_cells)))
    sorted_topic_ids = sorted([tid for tid in crosstab if tid != -1])
    if -1 in crosstab:
        sorted_topic_ids.append(-1)
    for tid in sorted_topic_ids:
        intents_dict = crosstab[tid]
        cells = " ".join(f"{intents_dict.get(i, 0):>14d}" for i in all_intents)
        label = "outliers" if tid == -1 else f"Topic {tid}"
        print(f"{label:<10} {cells}")


def alignment_summary(modeller):
    crosstab = modeller.cross_tab_against_intents()
    aligned_topics = 0
    total_topics = 0
    for tid, intents in crosstab.items():
        if tid == -1:
            continue
        total_topics += 1
        topic_total = sum(intents.values())
        if topic_total == 0:
            continue
        dominant_intent_count = max(intents.values())
        if dominant_intent_count / topic_total >= 0.7:
            aligned_topics += 1
    if total_topics == 0:
        print("\nNo topic clusters formed - try lowering min_topic_size.")
        return
    print(f"\nAlignment: {aligned_topics}/{total_topics} discovered topics are dominated "
          f"(>=70%) by a single hand-coded intent.")
    if aligned_topics == total_topics:
        print("Strong alignment - the unsupervised method recovers the taxonomy structure.")
    elif aligned_topics >= total_topics // 2:
        print("Partial alignment - taxonomy is broadly supported but some topics span intents.")
    else:
        print("Weak alignment - topics may be capturing axes other than intent.")


def main():
    if not CLEANED_PATH.exists():
        print(f"ERROR: cleaned file not found at {CLEANED_PATH}")
        print("Run methodology_demo/scripts/run_preprocessing.py first.")
        sys.exit(1)

    print("Loading cleaned contacts...")
    records = load_records(CLEANED_PATH)
    print(f"  {len(records)} records loaded")

    print("\nFitting BERTopic (downloads embedding model on first run)...")
    modeller = TopicModeller(min_topic_size=5, nr_topics=8)
    modeller.fit(records)

    print_topic_summary(modeller)
    print_crosstab(modeller)
    alignment_summary(modeller)

    print("\nDone.")


if __name__ == "__main__":
    main()