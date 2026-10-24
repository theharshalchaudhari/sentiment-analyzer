"""
Sentiment Analyzer
Classifies reviews as Positive, Negative or Neutral using NLTK's VADER
with a simple Streamlit web interface.

Run: streamlit run app.py
"""

import re

import nltk
import pandas as pd
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon once (cached on disk afterwards)
nltk.download("vader_lexicon", quiet=True)

st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

EMOJI = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}


# ------------------------------------------------------------------ helpers
@st.cache_resource
def load_analyzer():
    """Load the VADER model only once."""
    return SentimentIntensityAnalyzer()


def preprocess_text(text):
    """Remove URLs and extra spaces. Case and punctuation are kept because
    VADER uses them (e.g. 'GREAT!!!' is stronger than 'great')."""
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def get_label(compound):
    """Convert the compound score (-1 to +1) into a sentiment label."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_sentiment(analyzer, text):
    """Return the label and the VADER scores for one piece of text."""
    scores = analyzer.polarity_scores(preprocess_text(text))
    return get_label(scores["compound"]), scores


def show_label(label, compound):
    """Show the result in a coloured box."""
    message = f"{EMOJI[label]} {label}  (compound score: {compound:+.3f})"
    if label == "Positive":
        st.success(message)
    elif label == "Negative":
        st.error(message)
    else:
        st.info(message)


# ------------------------------------------------------------------ interface
analyzer = load_analyzer()

st.title("😊 Sentiment Analyzer")
st.caption("Find out whether a review is positive, negative or neutral (NLTK VADER)")

tab_single, tab_multi = st.tabs(["Single review", "Multiple reviews"])

# ---- Tab 1: one review
with tab_single:
    review = st.text_area(
        "Enter a review",
        height=150,
        placeholder="e.g. The movie was great and I enjoyed it a lot!",
    )
    if st.button("Analyze", type="primary", key="single"):
        if not review.strip():
            st.warning("Please enter some text first.")
        else:
            label, scores = analyze_sentiment(analyzer, review)
            show_label(label, scores["compound"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{scores['pos'] * 100:.0f}%")
            col2.metric("Neutral", f"{scores['neu'] * 100:.0f}%")
            col3.metric("Negative", f"{scores['neg'] * 100:.0f}%")

            st.bar_chart(pd.Series({
                "Positive": scores["pos"],
                "Neutral": scores["neu"],
                "Negative": scores["neg"],
            }))

# ---- Tab 2: many reviews (one per line)
with tab_multi:
    reviews_text = st.text_area(
        "Enter one review per line",
        height=200,
        key="multi_text",
        placeholder="The food was delicious\nThe service was terrible\nThe bill came at 9 pm",
    )
    if st.button("Analyze all", type="primary", key="multi"):
        reviews = [line for line in reviews_text.split("\n") if line.strip()]
        if not reviews:
            st.warning("Please enter at least one review.")
        else:
            rows = []
            for line in reviews:
                label, scores = analyze_sentiment(analyzer, line)
                rows.append({"Review": line, "Sentiment": label,
                             "Compound": scores["compound"]})
            df = pd.DataFrame(rows)

            st.dataframe(df)
            st.subheader("Summary")
            st.bar_chart(df["Sentiment"].value_counts())

st.divider()
st.caption("Built with Streamlit and NLTK | Sentiment Analyzer Mini Project")
