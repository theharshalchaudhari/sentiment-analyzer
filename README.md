# Sentiment Analyzer

A Streamlit web application that classifies text reviews as **Positive**, **Negative**, or **Neutral** using NLTK's VADER sentiment analyzer.

## Features

- Analyze a single review and view its sentiment score.
- Analyze multiple reviews, one per line.
- View positive, neutral, and negative score proportions.
- Compare sentiment labels across multiple reviews.

## Run locally

Requires Python 3.9 or newer.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app downloads the NLTK VADER lexicon on startup if it is not already available.

## Deploy

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Connect your GitHub account and select this repository.
3. Choose the `main` branch and set the app file path to `app.py`.
4. Deploy.

## Tech stack

- Python
- Streamlit
- NLTK VADER
- Pandas
