
"""
rule_based.py

Transparent keyword-and-pattern intent classifier.

Acts as the baseline classifier in the pipeline. Useful in production VoC for:
  - cold-start before sufficient labelled data exists for an ML model
  - fallback when the ML classifier confidence is low
  - explainability — a rule-based decision is auditable in a way an embedding
    nearest-neighbour decision is not
"""

import re
from typing import Dict, List, Optional, Tuple


INTENT_RULES = {
    "PRESCRIPTION_DELAY": [
        (r"\bwhere is my (prescription|order|medication|repeat)\b", 3),
        (r"\b(still waiting|not yet arrived|not arrived|not received|not come)\b", 3),
        (r"\b(running low|running out|run out|ran out)\b", 3),
        (r"\b(supposed to arrive|supposed to come|expected to arrive)\b", 2),
        (r"\b(days late|days overdue|weeks late)\b", 2),
        (r"\bdispatch\b", 1),
    ],
    "REPEAT_PRESCRIPTION_QUERY": [
        (r"\b(repeat list|repeat prescription)\b", 3),
        (r"\badd a new item\b", 3),
        (r"\badd an item\b", 3),
        (r"\bmissing from\b", 2),
        (r"\bmissing item\b", 2),
        (r"\bisn't on the list\b", 2),
        (r"\bisn't on my list\b", 2),
        (r"\bhow do i order\b", 2),
        (r"\bhow do i reorder\b", 2),
        (r"\bhow does the repeat work\b", 2),
        (r"\bhow does ordering work\b", 2),
        (r"\b(switching from|transferred from)\b", 1),
    ],
    "DELIVERY_ISSUE": [
        (r"\bwrong address\b", 3),
        (r"\bdelivered to the wrong\b", 3),
        (r"\b(damaged|soaked|soggy|opened|tampered)\b", 3),
        (r"\b(royal mail|rm) (card|note|notice)\b", 3),
        (r"\b(left in|left with|unsafe place|neighbour)\b", 2),
        (r"\b(returned to|sent back)\b", 1),
    ],
    "APP_WEBSITE_ISSUE": [
        (r"\bcan'?t log ?in\b", 3),
        (r"\bcannot log ?in\b", 3),
        (r"\bapp (is )?crashing\b", 3),
        (r"\bapp keeps crashing\b", 3),
        (r"\bpassword reset\b", 3),
        (r"\breset my password\b", 3),
        (r"\b2fa\b", 2),
        (r"\btwo factor\b", 2),
        (r"\btwo-factor\b", 2),
        (r"\bnhs app\b", 2),
        (r"\bnhs login\b", 2),
        (r"\b(reinstall|website|browser)\b", 1),
    ],
    "CLINICAL_QUERY": [
        (r"\bside effect\b", 3),
        (r"\binteraction\b", 3),
        (r"\binteract with\b", 3),
        (r"\bsafe to take\b", 3),
        (r"\bis it safe\b", 3),
        (r"\bsafe with\b", 3),
        (r"\bmissed a dose\b", 3),
        (r"\bskip a dose\b", 3),
        (r"\bdifferent (colour|shape|tablet|pill)\b", 2),
        (r"\bsame medication\b", 2),
        (r"\b(pregnant|pregnancy|breastfeeding|allergic)\b", 2),
        (r"\b(pharmacist|clinical|advice)\b", 1),
    ],
    "REFUND_BILLING": [
        (r"\bcharged twice\b", 3),
        (r"\bdouble charged\b", 3),
        (r"\bduplicate charge\b", 3),
        (r"\brefund\b", 3),
        (r"\bhc2\b", 3),
        (r"\bhc3\b", 3),
        (r"\bexemption certificate\b", 3),
        (r"\bprescription charge exempt\b", 3),
        (r"\bexempt from charges\b", 3),
        (r"\b(billed|invoice|payment taken)\b", 2),
        (r"\b(money back|cost|price)\b", 1),
    ],
    "GP_INTEGRATION": [
        (r"\bgp says\b", 3),
        (r"\bgp sent\b", 3),
        (r"\bgp surgery\b", 3),
        (r"\bgp practice\b", 3),
        (r"\bsurgery says\b", 3),
        (r"\bsurgery sent\b", 3),
        (r"\bnominat(e|ion|ed)\b", 3),
        (r"\bscript not sent\b", 2),
        (r"\bscript not received\b", 2),
        (r"\bblaming each other\b", 2),
        (r"\bstuck in the middle\b", 2),
    ],
    "CUSTOMER_SERVICE_RESPONSIVENESS": [
        (r"\b(third|3rd|second|2nd) (time|email|call)\b", 3),
        (r"\bno response\b", 3),
        (r"\bno reply\b", 3),
        (r"\bhaven'?t heard\b", 3),
        (r"\bhaven'?t replied\b", 3),
        (r"\bbeen on hold\b", 2),
        (r"\bbeen waiting\b", 2),
        (r"\bon hold for\b", 2),
        (r"\bround in circles\b", 2),
        (r"\bgetting nowhere\b", 2),
        (r"\bspeak to someone\b", 1),
        (r"\bspeak to a manager\b", 1),
        (r"\bescalate\b", 1),
    ],
}


_COMPILED_RULES = {
    intent: [(re.compile(pattern, re.IGNORECASE), weight)
             for pattern, weight in rules]
    for intent, rules in INTENT_RULES.items()
}


def score_intents(text: str) -> Dict[str, int]:
    """Compute a score per intent based on weighted pattern matches."""
    scores = {intent: 0 for intent in INTENT_RULES}
    for intent, compiled in _COMPILED_RULES.items():
        for pattern, weight in compiled:
            if pattern.search(text):
                scores[intent] += weight
    return scores


def classify(text: str, min_score: int = 2) -> Optional[Tuple[str, int]]:
    """Classify a single contact text into an intent.

    Returns:
        (intent_name, score) for the highest-scoring intent that clears
        min_score, or None if no intent clears the threshold.
    """
    scores = score_intents(text)
    top_intent = max(scores, key=scores.get)
    top_score = scores[top_intent]

    if top_score < min_score:
        return None
    return (top_intent, top_score)


def classify_with_confidence(text: str, min_score: int = 2) -> Dict:
    """Classify and return full diagnostic info — useful for the run script
    and for explainability in interview demonstrations."""
    scores = score_intents(text)
    top_intent = max(scores, key=scores.get)
    top_score = scores[top_intent]

    if top_score < min_score:
        return {
            "predicted_intent": None,
            "confidence": "none",
            "top_score": top_score,
            "all_scores": scores,
        }

    if top_score >= 5:
        confidence = "high"
    elif top_score >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "predicted_intent": top_intent,
        "confidence": confidence,
        "top_score": top_score,
        "all_scores": scores,
    }