import re
import spacy
import joblib
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from aspects import ASPECT_VOCAB, WORD_TO_ASPECT

# Words that signal positive/negative sentiment near an aspect keyword
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "perfect",
    "fantastic", "superb", "brilliant", "wonderful", "love", "best",
    "fast", "quick", "clear", "bright", "sharp", "beautiful", "nice",
    "reliable", "solid", "impressive", "reasonable", "worth", "happy"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "poor", "worst", "hate",
    "slow", "dead", "broken", "dim", "blurry", "ugly", "cheap",
    "expensive", "overpriced", "disappointing", "weak", "dreadful",
    "drains", "dies", "failed", "damaged", "delayed", "useless"
}

nlp            = spacy.load("en_core_web_sm")
embedder       = SentenceTransformer('all-MiniLM-L6-v2')
baseline_model = joblib.load('models/baseline_model.pkl')
xgb_model      = joblib.load('models/xgb_model.pkl')
le             = joblib.load('models/label_encoder.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_aspects(text):
    doc = nlp(text.lower())
    found = set()
    for chunk in doc.noun_chunks:
        for token in chunk:
            if token.text in WORD_TO_ASPECT:
                found.add(WORD_TO_ASPECT[token.text])
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and token.text in WORD_TO_ASPECT:
            found.add(WORD_TO_ASPECT[token.text])
    return list(found)

def split_into_sentences(text):
    return sent_tokenize(text)

def get_sentiment_for_sentence(sentence, use_upgrade=True):
    clean = clean_text(sentence)
    if use_upgrade:
        emb      = embedder.encode([clean])
        pred_enc = xgb_model.predict(emb)[0]
        label    = le.inverse_transform([pred_enc])[0]
        conf     = float(max(xgb_model.predict_proba(emb)[0]))
    else:
        label = baseline_model.predict([clean])[0]
        conf  = float(max(baseline_model.predict_proba([clean])[0]))
    return label, conf

def get_local_sentiment(sentence, aspect_keyword):
    """
    When a sentence contains multiple aspects, look at the words
    immediately surrounding the aspect keyword instead of classifying
    the whole sentence. Window of 4 words on each side.
    """
    words = sentence.lower().split()

    # Find position of the aspect keyword in the sentence
    keyword_pos = None
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^a-z]', '', word)
        if clean_word == aspect_keyword:
            keyword_pos = i
            break

    if keyword_pos is None:
        return None, None

    # Look at 4 words before and after the keyword
    window_start = max(0, keyword_pos - 4)
    window_end   = min(len(words), keyword_pos + 5)
    window_words = set(
        re.sub(r'[^a-z]', '', w) for w in words[window_start:window_end]
    )

    pos_hits = window_words & POSITIVE_WORDS
    neg_hits = window_words & NEGATIVE_WORDS

    if pos_hits and not neg_hits:
        return "positive", 0.80
    elif neg_hits and not pos_hits:
        return "negative", 0.80
    elif pos_hits and neg_hits:
        # Both present — trust whichever side is closer to the keyword
        return "neutral", 0.60
    else:
        return None, None  # no signal found in window

def analyze_review(review_text, use_upgrade=True):
    overall_label, overall_conf = get_sentiment_for_sentence(
        review_text, use_upgrade=use_upgrade
    )
    sentences = split_into_sentences(review_text)
    aspect_sentiments = {}

    for sentence in sentences:
        aspects_in_sentence = extract_aspects(sentence)

        if not aspects_in_sentence:
            continue

        if len(aspects_in_sentence) > 1:
            # Multiple aspects in one sentence — use local window per aspect
            for aspect in aspects_in_sentence:
                # Find which keyword triggered this aspect
                keyword_match = None
                for word in ASPECT_VOCAB[aspect]:
                    if word in sentence.lower():
                        keyword_match = word
                        break

                if keyword_match:
                    local_label, local_conf = get_local_sentiment(
                        sentence, keyword_match
                    )
                else:
                    local_label, local_conf = None, None

                # Fall back to full-sentence model if local window found nothing
                if local_label is None:
                    local_label, local_conf = get_sentiment_for_sentence(
                        sentence, use_upgrade=use_upgrade
                    )

                if aspect not in aspect_sentiments or \
                        local_conf > aspect_sentiments[aspect]['confidence']:
                    aspect_sentiments[aspect] = {
                        'sentiment':  local_label,
                        'confidence': local_conf,
                        'sentence':   sentence
                    }
        else:
            # Single aspect in sentence — full sentence model is reliable
            sent_label, sent_conf = get_sentiment_for_sentence(
                sentence, use_upgrade=use_upgrade
            )
            aspect = aspects_in_sentence[0]
            if aspect not in aspect_sentiments or \
                    sent_conf > aspect_sentiments[aspect]['confidence']:
                aspect_sentiments[aspect] = {
                    'sentiment':  sent_label,
                    'confidence': sent_conf,
                    'sentence':   sentence
                }

    return {
        'overall': {'sentiment': overall_label, 'confidence': overall_conf},
        'aspects': aspect_sentiments
    }