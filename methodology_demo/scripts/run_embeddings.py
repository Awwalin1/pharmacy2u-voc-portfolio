"""
run_embeddings.py

End-to-end demonstration of the embeddings + similarity search layer.

  1. Loads cleaned synthetic contacts.
  2. Embeds them with sentence-transformers (all-MiniLM-L6-v2).
  3. Picks four representative held-out queries (one per channel).
  4. For each query, prints the top-3 most similar historical contacts.

Run from project root:
    python methodology_demo/scripts/run_embeddings.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from methodology_demo.embeddings.embedder import ContactEmbedder
from methodology_demo.embeddings.similarity import top_k_similar


CLEANED_PATH = PROJECT_ROOT / "methodology_demo" / "data" / "cleaned_contacts.jsonl"


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_HEADER" in obj:
                continue
            records.append(obj)
    return records


def pick_one_per_channel(records):
    """Pick one example per channel as the held-out query set, return (query_set, corpus)."""
    seen, queries, corpus = set(), [], []
    for r in records:
        if r["channel"] not in seen:
            seen.add(r["channel"])
            queries.append(r)
        else:
            corpus.append(r)
    # Add the rest of the queries' channel back to the corpus so we still embed them
    # Actually — simpler: queries are held out, corpus is everyone else. That's the demo.
    return queries, corpus


def main():
    if not CLEANED_PATH.exists():
        print(f"ERROR: cleaned file not found at {CLEANED_PATH}")
        print("Run methodology_demo/scripts/run_preprocessing.py first.")
        sys.exit(1)

    print("Loading cleaned contacts...")
    records = load_records(CLEANED_PATH)
    queries, corpus = pick_one_per_channel(records)
    print(f"  {len(records)} total records → {len(queries)} held-out queries, {len(corpus)} corpus")

    print("\nLoading sentence-transformers model (first run downloads ~80MB)...")
    embedder = ContactEmbedder()

    print("Embedding corpus...")
    corpus_vectors = embedder.embed([r["cleaned_text"] for r in corpus])
    print(f"  Corpus matrix shape: {corpus_vectors.shape}")

    print("\nEmbedding queries and running similarity search...\n")
    for q in queries:
        q_vec = embedder.embed_one(q["cleaned_text"])
        top = top_k_similar(q_vec, corpus_vectors, corpus, k=3)

        print(f"━━━ QUERY [{q['channel']}] intent={q['intent']} ━━━")
        print(f"  text: {q['cleaned_text'][:110]}{'…' if len(q['cleaned_text']) > 110 else ''}\n")
        print("  Top 3 most similar in corpus:")
        for i, match in enumerate(top, 1):
            text = match["cleaned_text"][:100].replace("\n", " ")
            print(f"    {i}. sim={match['similarity']:.3f} "
                  f"[{match['channel']}/{match['intent']}] {text}{'…' if len(match['cleaned_text']) > 100 else ''}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()