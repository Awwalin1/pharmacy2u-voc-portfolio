"""
similarity.py

Cosine similarity search over a corpus of pre-embedded customer contacts.

Given a query (new contact) and a corpus (historical contacts), returns the
top-k most similar historical contacts. Foundational for:
  - retrieval-augmented chatbot intent matching
  - finding analogous past cases for new customer contacts
  - emerging issue detection (a new contact with no similar history is novel)
"""

from typing import List, Dict
import numpy as np


def top_k_similar(
    query_vector: np.ndarray,
    corpus_vectors: np.ndarray,
    corpus_records: List[Dict],
    k: int = 5,
) -> List[Dict]:
    """Find the top-k most similar contacts in the corpus to the query.

    Args:
        query_vector: (dim,) embedding of the query contact.
        corpus_vectors: (n, dim) embeddings of all historical contacts.
                        Assumed L2-normalised.
        corpus_records: list of n contact dicts, aligned with corpus_vectors.
        k: number of results to return.

    Returns:
        List of k dicts, each containing the original record fields plus a
        'similarity' score in [-1, 1] (1.0 = identical direction).
    """
    if corpus_vectors.shape[0] == 0:
        return []

    # Both query and corpus are L2-normalised, so dot product == cosine similarity
    scores = corpus_vectors @ query_vector  # shape (n,)

    # Top k indices, highest score first
    k = min(k, len(scores))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    results = []
    for idx in top_indices:
        record = dict(corpus_records[idx])
        record["similarity"] = float(scores[idx])
        results.append(record)
    return results