"""
test_channel_aware.py

Unit tests for the channel-aware preprocessing module.

Run from project root:
    python -m pytest methodology_demo/preprocessing/tests/ -v
"""

import pytest
from methodology_demo.preprocessing.channel_aware import (
    clean_phone,
    clean_email,
    clean_chat,
    clean_social,
    preprocess,
)


# ---------------------------------------------------------------------------
# Phone tests
# ---------------------------------------------------------------------------

class TestCleanPhone:
    def test_removes_um_uh_disfluencies(self):
        text = "yeah hi um I'm calling about er my prescription"
        result = clean_phone(text)
        assert "um" not in result.split()
        assert "er" not in result.split()
        assert "prescription" in result

    def test_removes_filler_phrases(self):
        text = "I sort of need help with like my order you know"
        result = clean_phone(text)
        assert "sort of" not in result
        assert "you know" not in result
        assert "help" in result
        assert "order" in result

    def test_strips_yeah_hi_opener(self):
        text = "yeah hi I need help with my repeat"
        result = clean_phone(text)
        assert not result.startswith("yeah hi")
        assert "repeat" in result

    def test_preserves_emotional_intensity(self):
        # Intensity words like 'really' and 'worried' carry signal — keep them
        text = "I'm really worried about my medication"
        result = clean_phone(text)
        assert "really" in result
        assert "worried" in result

    def test_lowercases_output(self):
        text = "YEAH HI UM I'M CALLING"
        result = clean_phone(text)
        assert result == result.lower()


# ---------------------------------------------------------------------------
# Email tests
# ---------------------------------------------------------------------------

class TestCleanEmail:
    def test_strips_dear_salutation(self):
        text = "Dear Pharmacy2U,\n\nI placed an order last week."
        result = clean_email(text)
        assert "Dear Pharmacy2U" not in result
        assert "placed an order" in result

    def test_strips_kind_regards_signoff(self):
        text = "I need a refund.\n\nKind regards,\nCustomer_A001"
        result = clean_email(text)
        assert "Kind regards" not in result
        assert "Customer_A001" not in result
        assert "refund" in result

    def test_strips_sent_from_iphone(self):
        text = "Where is my order.\n\nSent from my iPhone"
        result = clean_email(text)
        assert "Sent from my iPhone" not in result
        assert "Where is my order" in result

    def test_collapses_newlines(self):
        text = "Hi,\n\nLine one.\n\nLine two."
        result = clean_email(text)
        assert "\n" not in result


# ---------------------------------------------------------------------------
# Chat tests
# ---------------------------------------------------------------------------

class TestCleanChat:
    def test_reassembles_multiline_fragments(self):
        text = "still waiting on my repeat\nits been 8 days\nany update"
        result = clean_chat(text)
        assert "\n" not in result
        assert "still waiting" in result
        assert "any update" in result

    def test_expands_common_abbreviations(self):
        text = "pls help thx"
        result = clean_chat(text)
        assert "please" in result
        assert "thanks" in result

    def test_expands_dont_to_do_not(self):
        text = "i dont know whats happening"
        result = clean_chat(text)
        assert "do not" in result

    def test_expands_2fa(self):
        text = "2fa not working keeps timing out"
        result = clean_chat(text)
        assert "two factor authentication" in result

    def test_lowercases_output(self):
        text = "PLS HELP"
        result = clean_chat(text)
        assert result == result.lower()


# ---------------------------------------------------------------------------
# Social tests
# ---------------------------------------------------------------------------

class TestCleanSocial:
    def test_strips_at_mentions(self):
        text = "@Pharmacy2U your app keeps crashing"
        result = clean_social(text)
        assert "@Pharmacy2U" not in result
        assert "app keeps crashing" in result

    def test_keeps_hashtag_content(self):
        text = "really annoyed #pharmacyfail"
        result = clean_social(text)
        assert "pharmacyfail" in result
        assert "#" not in result

    def test_strips_urls(self):
        text = "see attached photo https://example.com/img.jpg disgraceful"
        result = clean_social(text)
        assert "https://" not in result
        assert "disgraceful" in result

    def test_preserves_sharper_tone(self):
        # Public-facing sharpness is itself signal — don't soften it
        text = "@Pharmacy2U this is unacceptable"
        result = clean_social(text)
        assert "unacceptable" in result


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_dispatches_phone_correctly(self):
        record = {"channel": "phone", "text": "yeah hi um I need help", "intent": "TEST"}
        result = preprocess(record)
        assert "cleaned_text" in result
        assert "um" not in result["cleaned_text"].split()

    def test_dispatches_email_correctly(self):
        record = {"channel": "email", "text": "Dear team,\n\nHelp me.\n\nRegards,\nA"}
        result = preprocess(record)
        assert "Dear team" not in result["cleaned_text"]
        assert "Regards" not in result["cleaned_text"]

    def test_passes_through_other_keys(self):
        record = {
            "channel": "chat",
            "text": "pls help",
            "intent": "PRESCRIPTION_DELAY",
            "vulnerability_flag": True,
        }
        result = preprocess(record)
        assert result["intent"] == "PRESCRIPTION_DELAY"
        assert result["vulnerability_flag"] is True

    def test_does_not_mutate_input(self):
        record = {"channel": "phone", "text": "yeah hi um help"}
        original_text = record["text"]
        preprocess(record)
        # input record should be unchanged
        assert record["text"] == original_text
        assert "cleaned_text" not in record

    def test_raises_on_unknown_channel(self):
        record = {"channel": "fax", "text": "hello"}
        with pytest.raises(ValueError, match="Unsupported channel"):
            preprocess(record)