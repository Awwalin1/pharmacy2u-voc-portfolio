"""
synthetic_contact_generator.py

DEMO ARTIFACT — generates 100 illustrative customer contact examples for
methodology demonstration purposes only.

This script does NOT use any real customer data. All examples are randomly
generated from templates designed to illustrate the linguistic differences
between contact channels (phone, email, chat, social) and the eight intent
categories defined in spec/intent_taxonomy.md.

No findings should be inferred from the generated dataset. The purpose is to
demonstrate the multi-channel pipeline architecture and intent taxonomy
applied in the methodology write-up.

Output: data/synthetic_contacts.jsonl  (one JSON record per line)

Run:    python synthetic_contact_generator.py
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)  # reproducible

# ---------------------------------------------------------------------------
# DISCLAIMER — written into every output file
# ---------------------------------------------------------------------------
DISCLAIMER = (
    "DEMO DATA — illustrative only. Randomly generated from templates to "
    "demonstrate methodology. Does not represent any real customer interactions. "
    "No findings should be inferred from this dataset."
)

# ---------------------------------------------------------------------------
# Placeholder identifiers — deliberately generic, not realistic
# ---------------------------------------------------------------------------
CUSTOMER_IDS = [f"Customer_{chr(65 + i)}{j:03d}" for i in range(8) for j in range(20)]
ORDER_REFS = [f"ORD-{n:06d}" for n in range(100000, 100200, 2)]
PRESCRIPTION_REFS = [f"RX-{n:05d}" for n in range(10000, 10200, 2)]

# ---------------------------------------------------------------------------
# Channel weights — phone and email dominate inbound contact in most VoC ops;
# chat and social smaller but growing. Weighted to reflect typical mix.
# ---------------------------------------------------------------------------
CHANNEL_WEIGHTS = {
    "phone": 0.40,
    "email": 0.30,
    "chat":  0.20,
    "social": 0.10,
}

# Intent weights — illustrative, weighted toward operational issues that
# typically dominate online pharmacy inbound contact.
INTENT_WEIGHTS = {
    "PRESCRIPTION_DELAY":            0.28,
    "REPEAT_PRESCRIPTION_QUERY":     0.16,
    "DELIVERY_ISSUE":                0.14,
    "APP_WEBSITE_ISSUE":             0.10,
    "CLINICAL_QUERY":                0.10,
    "REFUND_BILLING":                0.09,
    "GP_INTEGRATION":                0.08,
    "CUSTOMER_SERVICE_RESPONSIVENESS": 0.05,
}

# ---------------------------------------------------------------------------
# Templates per (channel, intent) — kept short, generic, channel-distinct.
# Phone: disfluencies, fragmentation, emotional opening
# Email: formal, complete sentences, often includes refs
# Chat: very short, abbreviations, fragmented
# Social: sharper, public-facing, often shorter
# ---------------------------------------------------------------------------

PHONE_TEMPLATES = {
    "PRESCRIPTION_DELAY": [
        "Yeah hi um I'm calling about my prescription it was supposed to be here days ago and I've got nothing and I'm running out so",
        "Hi there yeah I ordered my repeat I think it was last week and it still hasn't come through can you check {order_ref} please",
        "I'm getting really worried now I haven't had my medication for two days and I don't know what's happened to the order",
    ],
    "REPEAT_PRESCRIPTION_QUERY": [
        "Hi yeah I just wanted to add another item to my repeat list how do I do that",
        "So I'm new to the service and I'm trying to understand how the repeat thing works do I order each time or",
        "There's something missing from my repeat list it should have my inhaler on there but I can't see it",
    ],
    "DELIVERY_ISSUE": [
        "Yeah the parcel came but it's been left in the rain it's all soggy and I can't tell if the medication is alright",
        "Hi I'm calling because the delivery went to the wrong address again and the neighbour brought it round but it had been opened",
        "Royal Mail left a card but I was in the whole time nobody knocked",
    ],
    "APP_WEBSITE_ISSUE": [
        "I can't log in to the app it keeps saying my password is wrong but I just reset it",
        "The app keeps crashing every time I try to order I've tried reinstalling and everything",
        "Yeah hi I'm trying to nominate you on the NHS app but it's not letting me",
    ],
    "CLINICAL_QUERY": [
        "Hi I just got my new tablets and they're a different colour to before is that okay or should I worry",
        "I missed a dose this morning and I don't know whether to take two now or just skip it",
        "I'm pregnant and I just want to check if it's okay to keep taking these",
    ],
    "REFUND_BILLING": [
        "I've been charged twice for the same prescription can you sort that out please",
        "I'm exempt from charges but you've billed me anyway I sent the certificate over weeks ago",
        "Hi yeah I want a refund the order never came and you've taken the money",
    ],
    "GP_INTEGRATION": [
        "My GP says they've sent the script over but you're saying you haven't got it",
        "I want to nominate you guys as my pharmacy but I don't know how to do it through the surgery",
        "The surgery's blaming you and you're blaming them and I'm just trying to get my medication",
    ],
    "CUSTOMER_SERVICE_RESPONSIVENESS": [
        "This is the third time I've called about this and nobody's getting back to me",
        "I've sent two emails and had no reply I've been on hold for forty minutes",
        "I just want to speak to someone who can actually help me I keep going round in circles",
    ],
}

EMAIL_TEMPLATES = {
    "PRESCRIPTION_DELAY": [
        "Dear Pharmacy2U,\n\nI placed an order on {date} (reference {order_ref}) and I have not yet received my medication. I am running low and would appreciate an urgent update on dispatch.\n\nKind regards,\n{customer}",
        "Hello,\n\nMy repeat prescription was due to arrive last week. I have not received any dispatch confirmation. Could you please check on order {order_ref} and let me know when I can expect delivery.\n\nThank you,\n{customer}",
    ],
    "REPEAT_PRESCRIPTION_QUERY": [
        "Hi,\n\nI would like to add a new item to my repeat list. Please could you advise on the process for doing this through your service.\n\nRegards,\n{customer}",
        "Dear team,\n\nI recently transferred from a high street pharmacy and I am unclear on how often I need to reorder my medication on your platform. Could you please clarify.\n\nThanks,\n{customer}",
    ],
    "DELIVERY_ISSUE": [
        "Hello,\n\nMy parcel was delivered today but the packaging was open and one of the items appears to be missing. The order reference is {order_ref}. Please advise on next steps.\n\nRegards,\n{customer}",
        "Hi,\n\nA Royal Mail card was left at my address indicating a missed delivery, however I was at home throughout. The package has now been returned to the depot. Order ref {order_ref}.\n\nThanks,\n{customer}",
    ],
    "APP_WEBSITE_ISSUE": [
        "Hello,\n\nI have been unable to log in to my account for several days. The password reset link is not arriving in my inbox (I have checked spam). Could you please assist.\n\nRegards,\n{customer}",
        "Hi,\n\nYour app crashes every time I open the order screen on Android. I have reinstalled twice. My account email is on file.\n\nThanks,\n{customer}",
    ],
    "CLINICAL_QUERY": [
        "Hi,\n\nI received my new prescription today and the tablets look different to my previous batch (different shape and colour). Could a pharmacist please confirm whether this is the same medication.\n\nThank you,\n{customer}",
        "Dear pharmacist,\n\nI would like advice on whether it is safe to take my current medication alongside an over-the-counter painkiller. Please could someone get in touch.\n\nRegards,\n{customer}",
    ],
    "REFUND_BILLING": [
        "Hello,\n\nI have been charged twice for the same prescription on order {order_ref}. Please refund the duplicate charge at your earliest convenience.\n\nRegards,\n{customer}",
        "Hi,\n\nI hold a valid HC2 certificate but I have been charged the prescription fee on my last order. I attached the certificate when I registered.\n\nThanks,\n{customer}",
    ],
    "GP_INTEGRATION": [
        "Hi,\n\nMy GP surgery has confirmed that they sent my prescription to you on {date} but you do not appear to have it on your system. Could you please investigate.\n\nRegards,\n{customer}",
        "Hello,\n\nI would like to nominate Pharmacy2U as my dispenser. My GP has asked me to confirm this through your service. Please advise on the process.\n\nThanks,\n{customer}",
    ],
    "CUSTOMER_SERVICE_RESPONSIVENESS": [
        "Hello,\n\nThis is my third email regarding order {order_ref}. I have not had a substantive response to either of my previous messages. Please could someone with authority on this matter contact me.\n\nRegards,\n{customer}",
        "Hi,\n\nI have been trying to get hold of someone for over a week now via phone and email with no success. The original issue remains unresolved.\n\nThanks,\n{customer}",
    ],
}

CHAT_TEMPLATES = {
    "PRESCRIPTION_DELAY": [
        "hi where is my order {order_ref}",
        "still waiting on my repeat\nits been 8 days\nany update?",
        "order late\nrunning out of meds\nplease help",
    ],
    "REPEAT_PRESCRIPTION_QUERY": [
        "hi how do i add to my repeat list",
        "missing item from repeat\ncant see my inhaler",
        "new customer\nhow does ordering work",
    ],
    "DELIVERY_ISSUE": [
        "parcel arrived damaged\nbox open\n{order_ref}",
        "delivered to wrong house\nneighbour brought it",
        "rm card but i was home\nnobody knocked",
    ],
    "APP_WEBSITE_ISSUE": [
        "cant login\npw reset email not coming",
        "app crashes on order screen android",
        "2fa not working keeps timing out",
    ],
    "CLINICAL_QUERY": [
        "tablets look different this time\nsame medication?",
        "missed a dose what do i do",
        "is it ok with paracetamol",
    ],
    "REFUND_BILLING": [
        "charged twice for {order_ref}\nneed refund",
        "im exempt but charged anyway",
        "refund please order didnt arrive",
    ],
    "GP_INTEGRATION": [
        "gp says they sent script\nyou say you havent got it",
        "how do i nominate you",
        "surgery and you are blaming each other lol",
    ],
    "CUSTOMER_SERVICE_RESPONSIVENESS": [
        "third time asking about this\nstill no help",
        "no reply to emails\nbeen a week",
        "going round in circles need someone senior",
    ],
}

SOCIAL_TEMPLATES = {
    "PRESCRIPTION_DELAY": [
        "@Pharmacy2U order placed 10 days ago, still nothing. Running out of essential medication. Please respond.",
        "Anyone else having huge delays with @Pharmacy2U? My repeat is now over a week late and customer service hasn't replied.",
    ],
    "REPEAT_PRESCRIPTION_QUERY": [
        "@Pharmacy2U new to your service, can someone DM me about how to add items to a repeat list?",
        "Trying to switch my repeat over to @Pharmacy2U from my old pharmacy and the process is really unclear.",
    ],
    "DELIVERY_ISSUE": [
        "@Pharmacy2U parcel arrived completely soaked and the box was open. Photos attached. Order {order_ref}.",
        "Royal Mail allegedly attempted delivery while I was sat at home. @Pharmacy2U this keeps happening.",
    ],
    "APP_WEBSITE_ISSUE": [
        "@Pharmacy2U your app has crashed every time I've tried to order this week. Android. Anyone else?",
        "Can't log in to @Pharmacy2U for 3 days now. Password reset not arriving.",
    ],
    "CLINICAL_QUERY": [
        "@Pharmacy2U please can a pharmacist DM me, my new tablets look completely different to last month's batch.",
        "@Pharmacy2U is there a way to speak to a pharmacist directly without waiting on the phone for an hour?",
    ],
    "REFUND_BILLING": [
        "@Pharmacy2U charged me twice for the same prescription. Order {order_ref}. Sort it please.",
        "@Pharmacy2U I'm exempt from prescription charges and you've billed me. Sent the certificate twice.",
    ],
    "GP_INTEGRATION": [
        "@Pharmacy2U my GP confirms they've sent the script over and you're saying you don't have it. Stuck in the middle.",
        "@Pharmacy2U how do I formally nominate you as my pharmacy through the NHS app?",
    ],
    "CUSTOMER_SERVICE_RESPONSIVENESS": [
        "@Pharmacy2U third public message because no one is replying to emails or picking up phones. This is unacceptable.",
        "Genuinely impossible to get a response from @Pharmacy2U customer service. Two weeks and counting.",
    ],
}

CHANNEL_TEMPLATE_MAP = {
    "phone": PHONE_TEMPLATES,
    "email": EMAIL_TEMPLATES,
    "chat":  CHAT_TEMPLATES,
    "social": SOCIAL_TEMPLATES,
}

# ---------------------------------------------------------------------------
# Vulnerability and urgency cue insertion — a small fraction of records get
# additional cues appended that flip the boolean flags. Realistic enough to
# be useful for taxonomy demonstration; not so realistic that records read as
# real customer accounts.
# ---------------------------------------------------------------------------

VULNERABILITY_CUES = [
    " I'm 84 and I live alone.",
    " I have to mention I'm a full-time carer for my husband and I can't leave the house.",
    " I should say I have severe anxiety and this is making it worse.",
    " English is not my first language so please be patient.",
    " I'm on benefits and I can't afford to keep paying twice.",
]

URGENCY_CUES = [
    " I have literally one tablet left.",
    " I'm flying tomorrow and I need this for the trip.",
    " My surgery is on Monday and I need to start the medication today.",
    " I've been without it for two days now.",
]

def _maybe_add_cue(text: str, cue_pool: list, probability: float) -> tuple[str, bool]:
    """With given probability, append a cue from the pool. Return (text, flag)."""
    if random.random() < probability:
        return text + random.choice(cue_pool), True
    return text, False

# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

def _weighted_choice(weights: dict) -> str:
    keys = list(weights.keys())
    probs = list(weights.values())
    return random.choices(keys, weights=probs, k=1)[0]

def _fill_template(template: str, customer_id: str) -> str:
    return template.format(
        order_ref=random.choice(ORDER_REFS),
        prescription_ref=random.choice(PRESCRIPTION_REFS),
        customer=customer_id,
        date=(datetime.now() - timedelta(days=random.randint(2, 14))).strftime("%d/%m/%Y"),
    )

def generate_record() -> dict:
    channel = _weighted_choice(CHANNEL_WEIGHTS)
    intent = _weighted_choice(INTENT_WEIGHTS)
    customer_id = random.choice(CUSTOMER_IDS)
    template = random.choice(CHANNEL_TEMPLATE_MAP[channel][intent])
    text = _fill_template(template, customer_id)

    # Vulnerability cues are slightly more likely in CLINICAL_QUERY and
    # REFUND_BILLING; urgency cues more likely in PRESCRIPTION_DELAY and
    # CLINICAL_QUERY. Probabilities are illustrative.
    vuln_prob = 0.20 if intent in ("CLINICAL_QUERY", "REFUND_BILLING") else 0.10
    urg_prob  = 0.30 if intent in ("PRESCRIPTION_DELAY", "CLINICAL_QUERY") else 0.10

    text, vuln_flag = _maybe_add_cue(text, VULNERABILITY_CUES, vuln_prob)
    text, urg_flag  = _maybe_add_cue(text, URGENCY_CUES, urg_prob)

    return {
        "contact_id": str(uuid.uuid4()),
        "channel": channel,
        "intent": intent,
        "vulnerability_flag": vuln_flag,
        "urgency_flag": urg_flag,
        "customer_placeholder": customer_id,
        "text": text,
        "_disclaimer": "DEMO DATA — synthetic, illustrative only.",
    }

def main(n_records: int = 100, output_path: str = "data/synthetic_contacts.jsonl"):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        # First line is a disclaimer header in JSON form so any consumer
        # reading the file sees it before any data records.
        f.write(json.dumps({"_HEADER": DISCLAIMER}) + "\n")
        for _ in range(n_records):
            f.write(json.dumps(generate_record(), ensure_ascii=False) + "\n")

    print(f"Wrote {n_records} synthetic demo records to {out}")
    print(f"Disclaimer: {DISCLAIMER}")

if __name__ == "__main__":
    main()
