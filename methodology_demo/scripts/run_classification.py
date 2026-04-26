"""
run_classification.py

End-to-end demonstration of intent classification on the synthetic corpus.

  1. Loads cleaned synthetic contacts.
  2. Runs the rule-based classifier on each contact.
  3. Fits the nearest-neighbour classifier on a held-out training split,
     then predicts on the test split.
  4. Reports accuracy, agreement between the two classifiers, and shows
     example disagreements - the cases where the two approaches diverge,
     which are the most informative for understanding model behaviour.

Run from project root:
    python methodology_demo/scripts/run_classification.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from methodology_demo.classifier.rule_based import classify_with_confidence
from methodology_demo.classifier.nearest_neighbour import NearestNeighbourClassifier


CLEANED_PATH = PROJECT_ROOT / "methodology_demo" / "data" / "cleaned_contacts.jsonl"

random.seed(42)


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_HEADER" in obj:
                continue
            records.append(obj)
    return records


def split_train_test(records, test_fraction: float = 0.2):
    """Stratified-ish split - shuffle then split. With 100 records and 8
    intents this is good enough for a demo; production would use proper
    stratified splitting per intent."""
    shuffled = list(records)
    random.shuffle(shuffled)
    cutoff = int(len(shuffled) * (1 - test_fraction))
    return shuffled[:cutoff], shuffled[cutoff:]


def evaluate_rule_based(records):
    """Run rule-based classifier on every record and report accuracy."""
    correct = 0
    none_count = 0
    for r in records:
        result = classify_with_confidence(r["cleaned_text"])
        predicted = result["predicted_intent"]
        if predicted is None:
            none_count += 1
        elif predicted == r["intent"]:
            correct += 1
    total_classified = len(records) - none_count
    accuracy_when_classified = correct / total_classified if total_classified else 0.0
    return {
        "total": len(records),
        "correct": correct,
        "none_count": none_count,
        "accuracy_when_classified": accuracy_when_classified,
        "coverage": total_classified / len(records),
    }


def evaluate_knn(train, test, k: int = 5):
    """Fit k-NN on train, predict on test, report accuracy."""
    print(f"  Fitting k-NN on {len(train)} training records...")
    clf = NearestNeighbourClassifier(k=k)
    clf.fit(train)

    print(f"  Predicting on {len(test)} test records...")
    correct = 0
    predictions = []
    for r in test:
        result = clf.predict(r["cleaned_text"])
        predictions.append((r, result))
        if result["predicted_intent"] == r["intent"]:
            correct += 1
    return {
        "total": len(test),
        "correct": correct,
        "accuracy": correct / len(test),
        "predictions": predictions,
    }


def show_disagreements(test, knn_predictions, max_examples: int = 4):
    """Show cases where rule-based and k-NN classifiers disagree on test set.
    These are the most informative cases for understanding classifier
    behaviour."""
    disagreements = []
    for r, knn_result in knn_predictions:
        rb = classify_with_confidence(r["cleaned_text"])
        rb_pred = rb["predicted_intent"]
        knn_pred = knn_result["predicted_intent"]
        if rb_pred != knn_pred:
            disagreements.append((r, rb, knn_result))

    print(f"\nDisagreements between rule-based and k-NN: {len(disagreements)}/{len(test)}")
    for i, (r, rb, knn) in enumerate(disagreements[:max_examples], 1):
        print(f"\n--- Disagreement {i} ---")
        print(f"  text:          {r['cleaned_text'][:110]}")
        print(f"  true intent:   {r['intent']}")
        print(f"  rule-based:    {rb['predicted_intent']} (score={rb['top_score']})")
        print(f"  k-NN:          {knn['predicted_intent']} (conf={knn['confidence']:.2f})")


def main():
    if not CLEANED_PATH.exists():
        print(f"ERROR: cleaned file not found at {CLEANED_PATH}")
        print("Run methodology_demo/scripts/run_preprocessing.py first.")
        sys.exit(1)

    print("Loading cleaned contacts...")
    records = load_records(CLEANED_PATH)
    print(f"  {len(records)} records loaded")

    print("\n=== Rule-based classifier (full corpus) ===")
    rb_results = evaluate_rule_based(records)
    print(f"  Total:          {rb_results['total']}")
    print(f"  Correctly classified: {rb_results['correct']}")
    print(f"  Returned None:  {rb_results['none_count']} (no rule fired strongly enough)")
    print(f"  Coverage:       {rb_results['coverage']:.1%} of contacts received a prediction")
    print(f"  Accuracy:       {rb_results['accuracy_when_classified']:.1%} (of those classified)")

    print("\n=== k-NN classifier (train/test split) ===")
    train, test = split_train_test(records, test_fraction=0.2)
    print(f"  Train: {len(train)}, Test: {len(test)}")
    knn_results = evaluate_knn(train, test, k=5)
    print(f"  Accuracy: {knn_results['accuracy']:.1%} ({knn_results['correct']}/{knn_results['total']})")

    print("\n=== Cross-classifier disagreement analysis ===")
    show_disagreements(test, knn_results["predictions"], max_examples=4)

    print("\nDone.")


if __name__ == "__main__":
    main()