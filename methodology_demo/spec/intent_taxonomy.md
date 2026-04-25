# Intent Taxonomy — UK Online Pharmacy Customer Contact

> **DEMO ARTIFACT.** This taxonomy is illustrative and built to demonstrate methodology. It is not derived from Pharmacy2U internal data. Categories are inferred from public domain knowledge of UK online pharmacy operations and adapted from the dissertation taxonomy framework.

## Purpose

Single intent label per contact, applied across all channels (phone, email, chat, social). Designed to be the primary classification target for a Voice-of-Customer pipeline, with separate orthogonal flags for vulnerability and urgency.

## Eight Intent Categories

### 1. PRESCRIPTION_DELAY
Contact concerns a prescription that has not arrived in expected timeframe, is overdue, or where the customer is asking for a status update on dispatch.
- **Channel signal:** "where is my prescription," "still waiting," "supposed to arrive," "ran out of medication"
- **Edge cases:** GP-side delay (script not yet sent to pharmacy) vs pharmacy-side delay (script received but not dispatched) vs Royal Mail delay (dispatched but not delivered) — all classified as PRESCRIPTION_DELAY at intent level; root cause becomes a downstream tag.
- **Distinction from DELIVERY_ISSUE:** delay is about timing of an expected delivery; DELIVERY_ISSUE is about something going wrong with a delivery that did or should have happened.

### 2. REPEAT_PRESCRIPTION_QUERY
Contact concerns the repeat prescription process itself — ordering, frequency, what's on the repeat, switching items, missing items from the repeat list, or confusion about how the service works.
- **Channel signal:** "my repeat," "can I add," "why isn't X on my list," "how do I order again"
- **Edge cases:** First-time setup queries from new patients; queries about NHS vs private repeat handling; queries about switching from a high street pharmacy.
- **Distinction from CLINICAL_QUERY:** this is process-and-admin, not clinical advice.

### 3. DELIVERY_ISSUE
Contact concerns a problem with a delivery that was attempted, arrived damaged, was delivered to the wrong address, or was lost in transit.
- **Channel signal:** "didn't arrive," "wrong address," "damaged," "left in unsafe place," "Royal Mail card"
- **Edge cases:** Customer-side delivery preference issues (e.g. "please don't leave with neighbour") sit here too.
- **Distinction from PRESCRIPTION_DELAY:** something has happened to the delivery; it isn't simply running late.

### 4. APP_WEBSITE_ISSUE
Contact concerns the digital service itself — login problems, app crashes, password resets, ordering flow not working, account access.
- **Channel signal:** "can't log in," "app keeps crashing," "password reset not working," "can't find the order button"
- **Edge cases:** Two-factor authentication failures; NHS login linkage issues; queries from customers who can't see their repeat list in the app.

### 5. CLINICAL_QUERY
Contact requires pharmacist or clinical input — questions about medication, side effects, interactions, dosage, alternatives, generic switches, or anything where a clinical answer is needed.
- **Channel signal:** "is it safe to take with," "side effect," "different colour pill," "missed a dose," "pregnant"
- **Edge cases:** Should always trigger urgency review. Vulnerability flag often co-occurs.
- **Distinction from REPEAT_PRESCRIPTION_QUERY:** clinical questions need a pharmacist; repeat-process questions need an admin.

### 6. REFUND_BILLING
Contact concerns money — charged for something not received, double charged, prescription charge exemption queries, private prescription pricing, refund requests.
- **Channel signal:** "charged twice," "refund," "exempt from charges," "why was I billed"
- **Edge cases:** NHS exemption certificate queries (HC2/HC3); private prescription cost queries; pet prescription billing.

### 7. GP_INTEGRATION
Contact concerns the link between the customer's GP and the pharmacy — nominating Pharmacy2U, GP not sending the script, switching nomination, queries about what the GP has approved.
- **Channel signal:** "my GP," "nominate," "haven't sent," "surgery says they sent it"
- **Edge cases:** Often co-occurs with PRESCRIPTION_DELAY at root cause level. Classified by the customer's primary framing.

### 8. CUSTOMER_SERVICE_RESPONSIVENESS
Contact is itself a complaint about prior contact attempts — "I've called three times," "no one has replied to my email," "been on hold for an hour." This is a meta-category for contacts where the trigger is service responsiveness rather than a fresh underlying issue.
- **Channel signal:** "no response," "tried calling," "third time I've contacted," "still waiting to hear back"
- **Edge cases:** Should always tag the underlying issue too if identifiable, but intent label is CUSTOMER_SERVICE_RESPONSIVENESS to surface the responsiveness signal in aggregate reporting.

## Orthogonal Flags

### vulnerability_flag (boolean)
True where contact contains explicit vulnerability indicators: mentions of being elderly, alone, recently bereaved, mental health crisis, severe disability, carer dependency, language difficulty, financial hardship, safeguarding concerns. Always reviewed regardless of intent.

### urgency_flag (boolean)
True where contact indicates running out of medication imminently, clinical risk, time-bounded need (travel, surgery, appointment). Distinct from vulnerability — a customer can be urgent without being vulnerable, and vice versa.

## Notes on Channel Differences

The same intent surfaces differently across channels:
- **Phone:** Emotional, fragmented, often opens with frustration before stating the actual issue. Disfluencies present. Speaker turns matter.
- **Email:** More formal, often complete narrative, frequently includes prior reference numbers and full context.
- **Chat:** Shortest, most fragmented, multi-turn with delays between customer messages. Heavy use of abbreviations.
- **Social:** Public-facing, often sharper tone, sometimes performative for an audience, frequently shorter than email but more complete than chat.

A robust pipeline must apply channel-aware preprocessing before classification, not classify raw text directly.

## Out of Scope for This Demo

- Multi-intent labelling (single dominant intent only)
- Sub-categorisation within each intent (root cause tags would be a downstream layer)
- Sentiment scoring (orthogonal — sentiment is a separate axis)
- Resolution-state classification (open, in progress, resolved — operational, not VoC)
