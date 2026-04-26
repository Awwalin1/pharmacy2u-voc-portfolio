"""
test_rule_based.py

Unit tests for the rule-based intent classifier.

Run from project root:
    python -m pytest methodology_demo/classifier/tests/ -v
"""

import pytest
from methodology_demo.classifier.rule_based import (
    score_intents,
    classify,
    classify_with_confidence,
)


# ---------------------------------------------------------------------------
# Per-intent classification — minimal happy-path examples
# ---------------------------------------------------------------------------

class TestPrescriptionDelay:
    def test_classifies_classic_delay(self):
        result = classify("where is my prescription it has not arrived")
        assert result is not None
        assert result[0] == "PRESCRIPTION_DELAY"

    def test_classifies_running_low(self):
        result = classify("i am running low on my medication")
        assert result is not None
        assert result[0] == "PRESCRIPTION_DELAY"


class TestRepeatPrescriptionQuery:
    def test_classifies_add_item_to_repeat(self):
        result = classify("how do i add a new item to my repeat list")
        assert result is not None
        assert result[0] == "REPEAT_PRESCRIPTION_QUERY"


class TestDeliveryIssue:
    def test_classifies_wrong_address(self):
        result = classify("the parcel was delivered to the wrong address")
        assert result is not None
        assert result[0] == "DELIVERY_ISSUE"

    def test_classifies_damaged_package(self):
        result = classify("my parcel arrived damaged and soaked")
        assert result is not None
        assert result[0] == "DELIVERY_ISSUE"


class TestAppWebsiteIssue:
    def test_classifies_login_problem(self):
        result = classify("i cant login to the app")
        assert result is not None
        assert result[0] == "APP_WEBSITE_ISSUE"

    def test_classifies_password_reset(self):
        result = classify("password reset email not arriving")
        assert result is not None
        assert result[0] == "APP_WEBSITE_ISSUE"


class TestClinicalQuery:
    def test_classifies_safety_question(self):
        result = classify("is it safe to take with paracetamol")
        assert result is not None
        assert result[0] == "CLINICAL_QUERY"

    def test_classifies_missed_dose(self):
        result = classify("i missed a dose this morning what do i do")
        assert result is not None
        assert result[0] == "CLINICAL_QUERY"


class TestRefundBilling:
    def test_classifies_double_charge(self):
        result = classify("you charged twice for the same prescription i need a refund")
        assert result is not None
        assert result[0] == "REFUND_BILLING"

    def test_classifies_exemption(self):
        result = classify("i hold an hc2 certificate but you billed me anyway")
        assert result is not None
        assert result[0] == "REFUND_BILLING"


class TestGpIntegration:
    def test_classifies_gp_dispute(self):
        result = classify("my gp says they sent the script but you say you havent got it")
        assert result is not None
        assert result[0] == "GP_INTEGRATION"

    def test_classifies_nomination(self):
        result = classify("how do i nominate you as my pharmacy")
        assert result is not None
        assert result[0] == "GP_INTEGRATION"


class TestCustomerServiceResponsiveness:
    def test_classifies_repeat_contact(self):
        result = classify("this is the third time i have called and no response")
        assert result is not None
        assert result[0] == "CUSTOMER_SERVICE_RESPONSIVENESS"


# ---------------------------------------------------------------------------
# Threshold and edge-case behaviour
# ---------------------------------------------------------------------------

class TestThresholdBehaviour:
    def test_returns_none_for_unrelated_text(self):
        result = classify("hello how are you today")
        assert result is None

    def test_returns_none_below_min_score(self):
        # 'dispatch' alone scores 1, below default min_score=2
        result = classify("anything about dispatch", min_score=2)
        assert result is None

    def test_respects_custom_min_score(self):
        # Lower threshold should now accept the same text
        result = classify("anything about dispatch", min_score=1)
        assert result is not None
        assert result[0] == "PRESCRIPTION_DELAY"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestScoreIntents:
    def test_returns_score_for_every_intent(self):
        scores = score_intents("where is my prescription")
        # All eight intents should be present, even those with zero score
        assert len(scores) == 8
        assert "PRESCRIPTION_DELAY" in scores
        assert "REFUND_BILLING" in scores

    def test_unrelated_text_scores_zero_everywhere(self):
        scores = score_intents("the weather is nice today")
        assert all(score == 0 for score in scores.values())


class TestClassifyWithConfidence:
    def test_high_confidence_for_strong_match(self):
        result = classify_with_confidence(
            "where is my prescription it has not arrived running low"
        )
        assert result["predicted_intent"] == "PRESCRIPTION_DELAY"
        assert result["confidence"] in ("medium", "high")
        assert result["top_score"] >= 5

    def test_none_confidence_for_unmatched(self):
        result = classify_with_confidence("hello there")
        assert result["predicted_intent"] is None
        assert result["confidence"] == "none"

    def test_returns_all_scores(self):
        result = classify_with_confidence("where is my prescription")
        assert "all_scores" in result
        assert len(result["all_scores"]) == 8