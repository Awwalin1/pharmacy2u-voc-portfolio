# Multi-Channel Voice of Customer Pipeline - Methodology Demo

> **Demo artifact.** This repository demonstrates a multi-channel Voice of Customer (VoC) pipeline architecture for the UK online pharmacy sector. All contact data in this repository is **synthetic and illustrative only**, randomly generated from templates to demonstrate methodology. No findings should be inferred from the synthetic dataset, and the data does not represent any real customer interactions.

## Why this repository exists

This is a portfolio piece built alongside an application for the **Customer Insight Data Scientist** role at Pharmacy2U. The repository extends the methodology of my MSc dissertation - *Multi-Model Deep Learning Approach to Brand Sentiment Analysis for Real-Time Reputation Management* (University of Wolverhampton, 2025, Distinction), which applied a fine-tuned RoBERTa + BERT ensemble to 7,800 public Trustpilot reviews - into a multi-channel context that public review platforms do not capture.

Trustpilot data captures one channel: post-delivery public reviews. The Customer Insight Data Scientist role works across phone, email, chat and social channels, plus CRM-held contact and patient records. This repository is a methodology demonstration showing how that wider pipeline would be structured.

## What the JD asks for, and where each piece sits

The job description names eleven core responsibilities. The repository addresses them as follows:

| JD Responsibility | Module | Status |
| --- | --- | --- |
| Develop and apply NLP models for intent classification | methodology_demo/classifier/ | Working |
| Topic extraction (BERTopic) | methodology_demo/topics/ | Working |
| Sentiment analysis | methodology_demo/sentiment/ | Planned |
| Work with transformer-based models and text embeddings | methodology_demo/embeddings/ | Working |
| Build and maintain an end-to-end customer insights pipeline | methodology_demo/scripts/ | Working |
| Classify customer contacts by intent, topic, and root cause | methodology_demo/classifier/ plus methodology_demo/topics/ | Working |
| Track patient satisfaction signals across channels | Sentiment scoring layer | Planned |
| Analyse contact types suitable for self-service or proactive comms | Deflection analysis | Planned |
| Produce regular insight outputs | Streamlit dashboard | Planned |
| Define and maintain an intent taxonomy and query-pattern knowledge base | methodology_demo/spec/intent_taxonomy.md plus embedding similarity search | Working |
| Link customer contact data to CRM and patient records | SQLite mock + retention correlation | Planned |
| Working with messy, real-world text data | methodology_demo/preprocessing/ (channel-aware) | Working |
| Cloud ML platform exposure (beneficial) | Deployment artifacts | Planned |

The "Working" rows are functional with passing tests. The "Planned" rows are scoped in this README and being added across the build sequence.

## Repository structure

- methodology_demo/spec/intent_taxonomy.md
- methodology_demo/data/ (synthetic_contacts.jsonl, cleaned_contacts.jsonl)
- methodology_demo/scripts/ (synthetic_contact_generator, run_preprocessing, run_embeddings, run_classification, run_topics)
- methodology_demo/preprocessing/ (channel_aware.py + 23 passing tests)
- methodology_demo/embeddings/ (embedder.py, similarity.py)
- methodology_demo/classifier/ (rule_based.py, nearest_neighbour.py + 22 passing tests)
- methodology_demo/topics/ (bertopic_modeller.py)
- trustpilot/README.md (scope note)

## How the pipeline runs end-to-end

## Key design decisions

**Channel-aware preprocessing.** The same intent surfaces differently across channels - phone calls have disfluencies and emotional openers, emails have signatures and salutations, chat is fragmented and abbreviation-heavy, social posts have @-handles and hashtags. The module dispatches to channel-specific cleaners.

**Two complementary classifiers.** The rule-based classifier achieves 93.8% accuracy where it fires but only covers 80% of contacts. The k-NN classifier closes the coverage gap using semantic similarity over sentence-transformer embeddings. The run script reports disagreements between the two, which surface multi-intent contacts and edge cases worth human review.

**Supervised intents validated by unsupervised topics.** The BERTopic module clusters the same corpus without reference to the hand-coded intent labels, then the run script cross-tabulates discovered topics against intents. On the synthetic corpus, three of six discovered topics align cleanly with single intents while the others surface lexical superclusters that span multiple intents.

**Synthetic data with prominent disclaimers.** Every output file carries a top-of-file disclaimer header. The synthetic data demonstrates pipeline structure and taxonomy only.

## Test coverage

45 tests pass: 23 for channel-aware preprocessing, 22 for the rule-based classifier.

## What is intentionally not in this repository

**Real Pharmacy2U contact data.** Real-data validation of the underlying methodology is provided in the dissertation linked above.

**A Trustpilot scrape of Pharmacy2U public reviews.** See trustpilot/README.md for the scope note explaining the Cloudflare anti-bot constraint.

**Findings about Pharmacy2U operational reality.** All numbers in script outputs reflect performance on synthetic data only.

## About the author

Adewale Adekola - MSc Data Science (Distinction), University of Wolverhampton. Currently an Insight Data Scientist at CleverFolks.

