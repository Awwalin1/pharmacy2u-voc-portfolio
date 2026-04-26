"""
bertopic_modeller.py

BERTopic-based unsupervised topic modelling for the multi-channel
customer-contact corpus.

BERTopic uses sentence-transformer embeddings as input, applies UMAP for
dimensionality reduction, HDBSCAN for clustering, and class-based TF-IDF
to extract representative terms per topic. The result is a set of topics
discovered from the data without any reference to the hand-coded intent
taxonomy.

The interesting question this enables is: do the unsupervised topics line up
with the hand-coded intent categories in `methodology_demo/spec/intent_taxonomy.md`?
That cross-tabulation is the heart of the run script.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TopicModeller:
    """Wrapper around BERTopic configured for short customer-contact text.

    Defaults are chosen for a small corpus (around 100 records). For a larger
    real-world corpus, min_topic_size and the UMAP/HDBSCAN params would
    typically be retuned.
    """

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        min_topic_size: int = 5,
        nr_topics: Optional[int] = None,
    ):
        """
        Args:
            embedding_model_name: sentence-transformers model id.
            min_topic_size: minimum cluster size that will be considered a
                topic. With ~100 records, 5 is sensible; production with
                thousands of records would use 20-50.
            nr_topics: optional cap on number of topics. None lets HDBSCAN
                decide; passing an int merges similar topics down to that
                count. Useful for the demo to align with the 8-category
                taxonomy.
        """
        self.embedding_model_name = embedding_model_name
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self._embedding_model: Optional[SentenceTransformer] = None
        self._topic_model: Optional[BERTopic] = None
        self._topic_assignments: Optional[List[int]] = None
        self._fitted_records: Optional[List[Dict]] = None

    def _ensure_embedder(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def fit(self, records: List[Dict]) -> "TopicModeller":
        """Fit BERTopic on the cleaned text of each record.

        Args:
            records: list of dicts with at least 'cleaned_text' key.

        Returns:
            self, for chaining.
        """
        if not records:
            raise ValueError("fit() requires at least one record")

        texts = [r["cleaned_text"] for r in records]
        embedder = self._ensure_embedder()

        self._topic_model = BERTopic(
            embedding_model=embedder,
            min_topic_size=self.min_topic_size,
            nr_topics=self.nr_topics,
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _ = self._topic_model.fit_transform(texts)

        self._topic_assignments = list(topics)
        self._fitted_records = list(records)
        return self

    def topic_info(self):
        """Return BERTopic's standard topic info DataFrame.

        Columns include Topic, Count, Name, Representation (top words),
        Representative_Docs.
        """
        if self._topic_model is None:
            raise RuntimeError("call fit(records) before topic_info()")
        return self._topic_model.get_topic_info()

    def top_words_per_topic(self, n_words: int = 5) -> Dict[int, List[str]]:
        """Return the top-n representative words per discovered topic.

        Excludes topic -1 which BERTopic uses for outliers (records that
        did not fit any cluster well).
        """
        if self._topic_model is None:
            raise RuntimeError("call fit(records) before top_words_per_topic()")

        info = self._topic_model.get_topic_info()
        result = {}
        for topic_id in info["Topic"].tolist():
            if topic_id == -1:
                continue
            top_terms = self._topic_model.get_topic(topic_id)
            if not top_terms:
                continue
            result[topic_id] = [term for term, _weight in top_terms[:n_words]]
        return result

    def assignments(self) -> List[Tuple[Dict, int]]:
        """Pair each fitted record with its topic id (including -1 for outliers)."""
        if self._topic_assignments is None or self._fitted_records is None:
            raise RuntimeError("call fit(records) before assignments()")
        return list(zip(self._fitted_records, self._topic_assignments))

    def cross_tab_against_intents(self) -> Dict[int, Dict[str, int]]:
        """Cross-tabulate discovered topics against the hand-coded intent labels.

        Returns:
            {topic_id: {intent_name: count, ...}, ...}
            Useful for showing whether unsupervised discovery agrees with
            the supervised taxonomy.
        """
        from collections import defaultdict

        if self._topic_assignments is None or self._fitted_records is None:
            raise RuntimeError("call fit(records) before cross_tab_against_intents()")

        crosstab: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for record, topic_id in zip(self._fitted_records, self._topic_assignments):
            intent = record.get("intent", "UNKNOWN")
            crosstab[topic_id][intent] += 1

        # Convert to plain dicts for cleaner display
        return {tid: dict(intents) for tid, intents in crosstab.items()}