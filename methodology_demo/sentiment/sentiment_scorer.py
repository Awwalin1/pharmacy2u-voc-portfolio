"""
sentiment_scorer.py

Sentiment scoring for the multi-channel customer-contact corpus.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest - a RoBERTa model
fine-tuned on social media data, well-suited to short, informal customer
contact text. Three-class output: positive, neutral, negative.

For Pharmacy2U, sentiment is one of three signals worth tracking alongside
volume and intent. A spike in negative sentiment for PRESCRIPTION_DELAY
contacts in a given week is exactly the kind of leading indicator the JD
references when it talks about 'identify emerging issues early'.

The model is loaded lazily on first use to avoid ~500MB download at import.
"""

from typing import Dict, List, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class SentimentScorer:
    """RoBERTa-based sentiment scorer.

    Wraps a HuggingFace pipeline-style model with batch inference for
    efficiency on a corpus of contacts.
    """

    # Model output indices map to these labels. Order matters - it matches
    # the model's training output dimensions.
    LABELS = ["negative", "neutral", "positive"]

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.to(self.device)
            self._model.eval()

    def score(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """Score a list of texts. Returns one dict per input with:
            - label: the predicted class ('negative' | 'neutral' | 'positive')
            - confidence: float in [0, 1] for the predicted class
            - scores: dict of all three class probabilities
        """
        if not texts:
            return []

        self._ensure_loaded()
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for prob_row in probs:
                pred_idx = int(np.argmax(prob_row))
                results.append({
                    "label": self.LABELS[pred_idx],
                    "confidence": float(prob_row[pred_idx]),
                    "scores": {
                        self.LABELS[j]: float(prob_row[j])
                        for j in range(len(self.LABELS))
                    },
                })

        return results

    def score_one(self, text: str) -> Dict:
        """Score a single text. Returns the same dict shape as score()."""
        return self.score([text])[0]