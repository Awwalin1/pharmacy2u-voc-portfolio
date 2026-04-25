"""
channel_aware.py

Channel-aware preprocessing for multi-channel customer contacts.

Four contact channels (phone, email, chat, social) carry the same intents but
look completely different on the wire. A single preprocessing step would
either over-clean structured email or leave phone disfluencies in place. This
module dispatches to the right cleaner per channel.

Designed to operate on records produced by
methodology_demo/scripts/synthetic_contact_generator.py — channel field on
each record drives the dispatch.

Usage:
    from methodology_demo.preprocessing.channel_aware import preprocess
    cleaned = preprocess(record)   # record is a dict with 'channel' and 'text'
"""

import re
from typing import Dict


# ---------------------------------------------------------------------------
# Phone — disfluencies, fragmented speech, emotional opening
# ---------------------------------------------------------------------------

# Common spoken disfluencies that add noise without information
PHONE_DISFLUENCIES = [
    r"\b(um|uh|er|erm|ah|hmm)\b",
    r"\b(like|you know|i mean|sort of|kind of)\b",
    r"\byeah(\s+yeah)+\b",
]

# Greetings/openers that don't carry intent
PHONE_OPENERS = [
    r"^\s*(yeah\s+)?hi\s+(there\s+)?(yeah\s+)?",
    r"^\s*(yeah\s+)?hello\s+(there\s+)?(yeah\s+)?",
    r"^\s*so\s+",
]


def clean_phone(text: str) -> str:
    """Phone transcript cleaner. Removes disfluencies and trim openers, but
    preserves emotional intensity markers — they carry signal."""
    out = text.lower()
    for pattern in PHONE_DISFLUENCIES:
        out = re.sub(pattern, " ", out, flags=re.IGNORECASE)
    for pattern in PHONE_OPENERS:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Email — formal structure, signatures, salutations
# ---------------------------------------------------------------------------

EMAIL_SALUTATIONS = [
    r"^\s*(dear|hi|hello|hey)\s+[\w\s]+?[,\n]",
    r"^\s*good\s+(morning|afternoon|evening)[,\n]",
    r"^\s*to\s+whom\s+it\s+may\s+concern[,\n]",
]

EMAIL_SIGNOFFS = [
    r"\b(kind\s+regards|best\s+regards|regards|thanks|thank\s+you|sincerely|yours\s+(faithfully|sincerely)|cheers|best)[,\s]*\n.*$",
    r"\bsent\s+from\s+my\s+(iphone|ipad|android|samsung).*$",
]


def clean_email(text: str) -> str:
    """Email cleaner. Strips salutations, signoffs, and trailing signatures
    while preserving the body content that carries the actual intent."""
    out = text
    for pattern in EMAIL_SALUTATIONS:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE | re.MULTILINE)
    for pattern in EMAIL_SIGNOFFS:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE | re.DOTALL)
    out = re.sub(r"\n+", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Chat — fragmented, multi-line, abbreviations
# ---------------------------------------------------------------------------

# Common chat abbreviations that hurt classification if left in
CHAT_ABBREVIATIONS = {
    r"\bpls\b": "please",
    r"\bplz\b": "please",
    r"\bthx\b": "thanks",
    r"\btks\b": "thanks",
    r"\btia\b": "thanks in advance",
    r"\bdont\b": "do not",
    r"\bcant\b": "cannot",
    r"\bwont\b": "will not",
    r"\bim\b": "i am",
    r"\bive\b": "i have",
    r"\bid\b": "i would",
    r"\bpw\b": "password",
    r"\brm\b": "royal mail",
    r"\bgp\b": "gp",
    r"\b2fa\b": "two factor authentication",
}


def clean_chat(text: str) -> str:
    """Chat cleaner. Reassembles multi-line fragments into a single utterance
    and expands common abbreviations into full words for downstream models."""
    # Reassemble fragments — chat messages often arrive as multiple short lines
    out = re.sub(r"\n+", " ", text.lower())

    # Expand abbreviations
    for pattern, replacement in CHAT_ABBREVIATIONS.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    out = re.sub(r"\s+", " ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Social — @-handles, hashtags, link wrappers
# ---------------------------------------------------------------------------

def clean_social(text: str) -> str:
    """Social cleaner. Strips @-handles (no signal for intent classification)
    and normalises hashtags into plain words. Preserves the rest, including
    sharper public-facing tone which is itself signal for VoC reporting."""
    out = text
    out = re.sub(r"@\w+", "", out)               # strip @mentions
    out = re.sub(r"#(\w+)", r"\1", out)          # keep hashtag content, drop #
    out = re.sub(r"https?://\S+", "", out)       # strip URLs
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

CHANNEL_DISPATCH = {
    "phone": clean_phone,
    "email": clean_email,
    "chat": clean_chat,
    "social": clean_social,
}


def preprocess(record: Dict) -> Dict:
    """Channel-aware preprocessing dispatcher.

    Args:
        record: dict with at minimum 'channel' (str) and 'text' (str) keys.
                Other keys (intent, flags, etc.) are passed through unchanged.

    Returns:
        New dict with the same structure plus a 'cleaned_text' key containing
        the channel-appropriate preprocessed version.

    Raises:
        ValueError: if channel is not one of the four supported channels.
    """
    channel = record.get("channel")
    text = record.get("text", "")

    if channel not in CHANNEL_DISPATCH:
        raise ValueError(
            f"Unsupported channel '{channel}'. "
            f"Expected one of: {sorted(CHANNEL_DISPATCH.keys())}"
        )

    cleaner = CHANNEL_DISPATCH[channel]
    out = dict(record)  # shallow copy so we don't mutate input
    out["cleaned_text"] = cleaner(text)
    return out