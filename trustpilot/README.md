# Trustpilot Public Review Analysis — Scope Note

## Original plan

This folder was scoped to hold a Pharmacy2U public Trustpilot review scrape and analysis pipeline, applying the dissertation methodology (BeautifulSoup + transformer-based intent and sentiment classification) to a domain-relevant dataset of 5,000–10,000 recent reviews.

## Why the public scrape is not included in this repo

Trustpilot's anti-bot infrastructure has been substantially hardened since the dissertation work was completed in early 2025. Direct scraping via `requests` + `BeautifulSoup`, headless Playwright, and `cloudscraper` were each tested in this build and all returned 403 responses or were blocked at the Cloudflare challenge layer. Reliable scraping at the volume needed would require either a paid third-party service (e.g. ScrapingBee, Bright Data) or residential proxy rotation — both reasonable production solutions, but out of scope for a personal portfolio build.

## Where the equivalent real-data work exists

Real-data validation of the underlying methodology is provided in my MSc dissertation: *Multi-Model Deep Learning Approach to Brand Sentiment Analysis for Real-Time Reputation Management* (University of Wolverhampton, 2025, Distinction). That work applied the same transformer-based pipeline (RoBERTa fine-tuning, BERT/RoBERTa weighted ensemble, BERTopic, automated department-level taxonomy) to **7,800 public Trustpilot reviews** across three brands.

The methodology demonstrated in this repository's `methodology_demo/` is an extension of that work into a multi-channel context (phone, email, chat, social) that public review platforms do not capture.