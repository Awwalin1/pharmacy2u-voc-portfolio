# Multi-Channel Voice of Customer Pipeline - Methodology Demo

> **Demo artifact.** This repository demonstrates a multi-channel Voice of Customer (VoC) pipeline architecture for the UK online pharmacy sector. All contact data in this repository is **synthetic and illustrative only**, randomly generated from templates to demonstrate methodology. No findings should be inferred from the synthetic dataset, and the data does not represent any real customer interactions.

## Why this repository exists

This is a portfolio piece built alongside an application for the **Customer Insight Data Scientist** role at Pharmacy2U. The repository extends the methodology of my MSc dissertation - *Multi-Model Deep Learning Approach to Brand Sentiment Analysis for Real-Time Reputation Management* (University of Wolverhampton, 2025, Distinction), which applied a fine-tuned RoBERTa + BERT ensemble to 7,800 public Trustpilot reviews - into a multi-channel context that public review platforms do not capture.

Trustpilot data captures one channel: post-delivery public reviews. The Customer Insight Data Scientist role works across phone, email, chat and social channels, plus CRM-held contact and patient records. This repository is a methodology demonstration showing how that wider pipeline would be structured.

## What the JD asks for, and where each piece sits

The job description names eleven core responsibilities. The repository addresses them as follows:

| JD Responsibility | Module | Status |
| --- | --- | --- |
| Develop and apply NLP models for intent classification | `methodology_demo/classifier/` | Working |
| Work with transformer-based models and text embeddings | `methodology_demo/embeddings/` | Working |
| Build and maintain an end-to-end customer insights pipeline | `methodology_demo/scripts/` | Working |
| Classify customer contacts by intent, topic, and root cause | `methodology_demo/classifier/` (rule-based + k-NN ensemble) | Working |
| Track patient satisfaction signals across channels | Sentiment scoring layer | In progress |
| Analyse contact types suitable for self-service or proactive comms | Deflection analysis | Planned |
| Produce regular insight outputs | Streamlit dashboard | Planned |
| Define and maintain an intent taxonomy and query-pattern knowledge base | `methodology_demo/spec/intent_taxonomy.md` | Working |
| Link customer contact data to CRM and patient records | Architecture diagram + schema | Planned |
| Working with messy, real-world text data | `methodology_demo/preprocessing/` (channel-aware) | Working |
| Cloud ML platform exposure | Deployment artifacts | Planned |

The "Working" rows are functional with passing tests. The "Planned" rows are scoped in this README and will be added across the coming evenings.

## Repository structure

## How the pipeline runs end-to-end

Three scripts run in sequence, each consuming the previous one's output:

```bash
# 1. Generate synthetic contacts (writes data/synthetic_contacts.jsonl)
python methodology_demo/scripts/synthetic_contact_generator.py

# 2. Apply channel-aware preprocessing (writes data/cleaned_contacts.jsonl)
python methodology_demo/scripts/run_preprocessing.py

# 3. Run intent classification (rule-based + k-NN with disagreement analysis)
python methodology_demo/scripts/run_classification.py

# Optional: similarity search demo
python methodology_demo/scripts/run_embeddings.py
```

## Key design decisions

**Channel-aware preprocessing.** The same intent surfaces differently across channels - phone calls have disfluencies and emotional openers, emails have signatures and salutations, chat is fragmented and abbreviation-heavy, social posts have @-handles and hashtags. A single preprocessing step would either over-clean structured email or leave phone disfluencies in place. The module dispatches to channel-specific cleaners.

**Two complementary classifiers.** The rule-based classifier achieves 93.8% accuracy where it fires but only covers 80% of contacts (the rest fall below the confidence threshold and are returned as `None` - "needs human triage"). The k-NN classifier closes the coverage gap using semantic similarity over sentence-transformer embeddings. The run script reports disagreements between the two, which surface multi-intent contacts and edge cases worth human review. In production this would be ensembled: rule-based as the high-precision layer, k-NN as the recall layer for unmatched contacts.

**Synthetic data with prominent disclaimers.** Every output file carries a top-of-file disclaimer header. The synthetic data demonstrates pipeline structure and taxonomy - it is not used to train models, no findings are reported from it, and placeholder identifiers (`Customer_A001` etc.) are deliberately non-realistic.

## Test coverage

```bash
python -m pytest methodology_demo/ -v
```

45 tests pass: 23 covering channel-aware preprocessing, 22 covering the rule-based classifier across all eight intents, threshold behaviour, and diagnostic confidence scoring.

## What is intentionally not in this repository

**Real Pharmacy2U contact data.** The role works with internal CRM-held contacts; this repository works with synthetic illustrative data. Real-data validation of the underlying methodology is provided in the dissertation (linked above).

**A Trustpilot scrape of Pharmacy2U public reviews.** Originally scoped, found to be blocked by Cloudflare on every approach attempted (`requests`, headless Playwright, `cloudscraper`). See `trustpilot/README.md` for the full scope note. A paid scraping service like ScrapingBee would resolve this in production but is out of scope for a personal portfolio build.

**Findings about Pharmacy2U operational reality.** The classifier accuracy and similarity scores reported in script outputs reflect performance on synthetic data only. They demonstrate that the methodology runs end-to-end; they do not characterise real customer contact patterns.

