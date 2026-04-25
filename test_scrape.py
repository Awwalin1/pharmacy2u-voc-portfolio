import cloudscraper
from bs4 import BeautifulSoup

URL = "https://uk.trustpilot.com/review/www.pharmacy2u.co.uk"

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "desktop": True}
)

response = scraper.get(URL, timeout=20)

print(f"Status code: {response.status_code}")
print(f"Response length: {len(response.text)} characters")

soup = BeautifulSoup(response.text, "lxml")
title = soup.find("title")
print(f"Page title: {title.text.strip() if title else 'NONE FOUND'}")

# Look for review-specific elements
review_articles = soup.find_all("article")
print(f"<article> elements found: {len(review_articles)}")

# Sanity check — look for known Pharmacy2U review markers
checks = {
    "pharmacy2u in page": "pharmacy2u" in response.text.lower(),
    "trustscore present": "trustscore" in response.text.lower(),
    "reviews present": "reviews" in response.text.lower(),
}
for k, v in checks.items():
    print(f"  {k}: {v}")

if response.status_code == 200 and review_articles:
    print("\nSTATUS: SUCCESS — page loaded and review elements present")
elif response.status_code == 200:
    print("\nSTATUS: Got page but no review articles — selectors may need updating")
else:
    print(f"\nSTATUS: Failed with status {response.status_code}")