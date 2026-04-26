"""
nearest_neighbour.py

Embedding-based nearest-neighbour intent classifier.

Uses the embeddings module to find the k most similar contacts in a labelled
corpus and predicts intent by majority vote among the neighbours, weighted
by similarity score. Complements the rule-based classifier:

  - rule-based handles cases where keywords are present and unambiguous
  - nearest-neighbour handles paraphrased or novel phrasing the rules miss
  - the run script shows agreement and disagreement between the two

This is the classifier that benefits most from semantic similarity - it does
not need exact keyword matches, only similar meaning to known examples.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

from methodology_demo.embeddings.embedder import ContactEmbedder
from methodology_demo.embeddings.similarity import top_k_similar


class NearestNeighbourClassifier:
    """k-NN classifier over sentence-transformer embeddings.

    Fitting stores the corpus and embeddings; prediction finds the k nearest
    labelled contacts and votes among them, weighted by cosine similarity.
    """

    def __init__(self, k: int = 5, embedder: Optional[ContactEmbedder] = None):
        self.k = k
        self.embedder = embedder if embedder is not None else ContactEmbedder()
        self._corpus_records: List[Dict] = []
        self._corpus_vectors: Optional[np.ndarray] = None

    def fit(self, records: List[Dict]) -> None:
        """Store labelled corpus records and compute their embeddings.

        Args:
            records: list of dicts with at minimum 'cleaned_text' and 'intent' keys.
        """
        if not records:
            raise ValueError("fit() requires at least one record")

        self._corpus_records = list(records)
        texts = [r["cleaned_text"] for r in records]
        self._corpus_vectors = self.embedder.embed(texts)

    def predict(self, text: str) -> Dict:
        """Predict intent for a new contact text.

        Returns:
            Dict with keys:
              - predicted_intent: str
              - confidence: float in [0, 1] (sum of weighted votes for predicted)
              - neighbours: list of the k nearest neighbour records with similarity
        """
        if self._corpus_vectors is None or len(self._corpus_records) == 0:
            raise RuntimeError("classifier has not been fit - call fit(records) first")

        query_vec = self.embedder.embed_one(text)
        neighbours = top_k_similar(
            query_vec, self._corpus_vectors, self._corpus_records, k=self.k
        )

        # Weighted vote - each neighbour's similarity is its vote weight
        votes: Dict[str, float] = defaultdict(float)
        total_weight = 0.0
        for n in neighbours:
            sim = n["similarity"]
            # Only count positive similarities - negative would mean opposite direction
            if sim > 0:
                votes[n["intent"]] += sim
                total_weight += sim

        if total_weight == 0:
            return {
                "predicted_intent": None,
                "confidence": 0.0,
                "neighbours": neighbours,
            }

        predicted_intent = max(votes, key=votes.get)
        confidence = votes[predicted_intent] / total_weight

        return {
            "predicted_intent": predicted_intent,
            "confidence": float(confidence),
            "neighbours": neighbours,
        }