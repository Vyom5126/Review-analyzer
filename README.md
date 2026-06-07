# Aspect-Based Sentiment Analyzer

Most sentiment analyzers tell you if a review is positive or negative.
This one tells you *what specifically* the reviewer liked or didn't like.


---

## The problem with overall sentiment

Take this review:

> "The battery life is terrible but the screen looks absolutely amazing.
> The price is reasonable for what you get."

A normal sentiment classifier might call this **neutral** — the good and bad
cancel out. That's not useful to anyone.

This project breaks it down:

| Aspect   | Sentiment | Evidence |
|----------|-----------|----------|
| Battery  | ❌ Negative | "The battery life is terrible" |
| Screen   | ✅ Positive | "the screen looks absolutely amazing" |
| Price    | ✅ Positive | "The price is reasonable" |

A 3-star review was hiding two very different stories.
That's the signal a product team actually needs.

---

## What it does

1. Takes any product review as input
2. Classifies the overall sentiment (positive / neutral / negative)
3. Detects which product aspects are mentioned (battery, screen, camera,
   price, delivery)
4. Classifies sentiment separately for each aspect

---

## How it works

**Step 1 — Sentiment Classification**

Two models were built and compared:

| Model | Macro F1 | Notes |
|-------|----------|-------|
| TF-IDF + Logistic Regression | 0.6004 | Baseline - fast, uses `class_weight='balanced'` |
| Sentence-Transformers + XGBoost | 0.6767 | Upgraded - +12.7% over baseline |

The baseline converts words to numbers by frequency (TF-IDF) and uses
`class_weight='balanced'` to handle the skewed dataset (80% positive reviews).

The upgrade encodes each sentence into a 384-dimensional vector that represents
its meaning, so "this is garbage" and "this is terrible" land close together
even though they share no words. XGBoost is then trained on those vectors with
`sample_weight='balanced'` — matching the same class-imbalance correction the
baseline uses.

The biggest gain is on the **neutral** class: the first (unbalanced) XGBoost
attempt scored F1=0.02 on neutral reviews, almost always predicting positive
instead. With balanced sample weights that rises to F1=0.42, which pulls the
macro average above the baseline.

**Step 2 — Aspect Extraction**

Uses spaCy to find nouns in the review, then matches them against a
hand-crafted vocabulary of 5 aspects and their synonyms.
No labeled data needed.

**Step 3 — Aspect-Level Sentiment**

Splits the review into individual sentences.
For each sentence, finds which aspect it mentions, then runs the
sentiment classifier on that sentence alone not the full review.
When multiple aspects appear in the same sentence, a local context
window around each keyword is used instead.

---

## Run it locally

```bash
git clone https://github.com/Vyom5126/Review-analyzer
cd Review-analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Tech stack

- **scikit-learn** - TF-IDF vectorizer and Logistic Regression
- **sentence-transformers** - semantic text embeddings
- **XGBoost** - gradient boosting classifier
- **spaCy** - noun extraction and dependency parsing
- **NLTK** - sentence tokenization
- **Streamlit** - web app and deployment
- **Hugging Face datasets** - Amazon Electronics reviews (streamed, not downloaded)

---

## Dataset

Amazon Electronics Reviews from the
[McAuley Lab Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
dataset. 50,000 reviews streamed directly — no 22GB download needed.

Labels were created from star ratings:
1-2 stars → negative, 3 stars → neutral, 4-5 stars → positive.
No manual labeling required.

---

## Limitations

- Only tracks 5 aspects: battery, screen, camera, price, delivery
- Works best on electronics product reviews
- Aspect coverage is around 23% as most reviews don't mention specific features
- Negation handling is basic ("not bad" may still read as negative)

See [APPROACH.md](APPROACH.md) for the full engineering breakdown,
what didn't work, and what I'd do differently.

---
