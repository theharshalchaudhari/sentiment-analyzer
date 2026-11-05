"""
Sentiment Analyzer — glassmorphism UI
Run with: streamlit run app.py
"""

import io
import re
import html
from typing import Dict, List, Tuple

import nltk
import pandas as pd
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer


st.set_page_config(
    page_title="Sentiment Studio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------- Design system -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --ink: #172033;
      --muted: #758096;
      --line: rgba(255,255,255,.78);
      --glass: rgba(255,255,255,.66);
      --blue: #526df5;
      --blue-dark: #4059df;
      --green: #159b72;
      --red: #df5b68;
      --amber: #b98228;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
      background:
        radial-gradient(ellipse at 8% 3%, rgba(151,177,255,.34), transparent 34%),
        radial-gradient(ellipse at 94% 12%, rgba(205,185,255,.34), transparent 30%),
        radial-gradient(ellipse at 52% 100%, rgba(166,221,255,.22), transparent 34%),
        linear-gradient(135deg, #f5f7fc 0%, #eef2fa 48%, #f7f5fc 100%);
      color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1.5rem; }
    .block-container { max-width: 1180px; padding: 2.2rem 2.1rem 4rem; }
    #MainMenu, footer { visibility: hidden; }
    .hero {
      position: relative; overflow: hidden;
      padding: 30px 34px 28px; border-radius: 28px;
      background: linear-gradient(115deg, rgba(255,255,255,.82), rgba(255,255,255,.48));
      border: 1px solid rgba(255,255,255,.92);
      box-shadow: 0 18px 55px rgba(49,65,111,.09), inset 0 1px 0 #fff;
      backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
      margin-bottom: 24px;
    }
    .hero:after {
      content:""; position:absolute; width:270px; height:270px; border-radius:50%;
      right:-72px; top:-145px;
      background: radial-gradient(circle, rgba(119,145,255,.24), rgba(119,145,255,0));
      pointer-events:none;
    }
    .brand-row { display:flex; align-items:center; gap:11px; margin-bottom:24px; }
    .brand-mark {
      width:38px; height:38px; display:grid; place-items:center; border-radius:13px;
      color:white; background:linear-gradient(145deg,#7189ff,#4c62e8);
      box-shadow:0 7px 18px rgba(82,109,245,.28);
    }
    .brand-name { font:800 14px 'Manrope',sans-serif; letter-spacing:.02em; color:#29334a; }
    .eyebrow {
      color:#6678db; font-size:11px; font-weight:700; letter-spacing:.15em;
      text-transform:uppercase; margin-bottom:9px;
    }
    .hero h1 {
      font:800 clamp(30px,4vw,43px)/1.12 'Manrope',sans-serif;
      letter-spacing:-.045em; color:#172033; margin:0 0 10px;
    }
    .hero p { max-width:590px; color:#68748b; font-size:14px; line-height:1.7; margin:0; }
    .hero-meta { display:flex; flex-wrap:wrap; gap:9px; margin-top:20px; }
    .meta-chip {
      display:inline-flex; align-items:center; gap:7px; padding:7px 11px;
      border-radius:99px; background:rgba(255,255,255,.67);
      border:1px solid rgba(255,255,255,.95); color:#667187;
      font-size:11px; font-weight:600;
    }
    .glass-card {
      background:linear-gradient(145deg,rgba(255,255,255,.82),rgba(255,255,255,.54));
      border:1px solid rgba(255,255,255,.94); border-radius:23px;
      box-shadow:0 12px 35px rgba(48,63,105,.065),inset 0 1px 0 rgba(255,255,255,.9);
      backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
      padding:22px 24px; margin-bottom:16px;
    }
    .section-kicker {
      color:#8b95a8; text-transform:uppercase; letter-spacing:.13em;
      font-size:10px; font-weight:700; margin:0 0 7px;
    }
    .section-title {
      color:#202b42; font:750 19px 'Manrope',sans-serif; letter-spacing:-.025em;
      margin:0 0 5px;
    }
    .section-copy { color:#8490a4; font-size:12px; margin:0 0 16px; }
    .stTabs [data-baseweb="tab-list"] {
      gap:7px; background:rgba(255,255,255,.46); border:1px solid rgba(255,255,255,.8);
      padding:6px; border-radius:15px; width:max-content; margin-bottom:16px;
    }
    .stTabs [data-baseweb="tab"] {
      height:38px; border-radius:10px; color:#7c879b; padding:0 17px;
      font-size:12px; font-weight:600; border:0;
    }
    .stTabs [aria-selected="true"] {
      background:rgba(255,255,255,.94) !important; color:#4059df !important;
      box-shadow:0 3px 12px rgba(43,57,100,.09);
    }
    .stTabs [data-baseweb="tab-highlight"] { display:none; }
    .stTextArea textarea, .stTextInput input {
      background:rgba(255,255,255,.74) !important; color:#263149 !important;
      border:1px solid rgba(213,220,236,.9) !important; border-radius:14px !important;
      box-shadow:inset 0 1px 3px rgba(38,52,88,.025) !important;
      font-size:13px !important; line-height:1.6 !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
      border-color:rgba(82,109,245,.65) !important;
      box-shadow:0 0 0 3px rgba(82,109,245,.10) !important;
    }
    label, [data-testid="stFileUploader"] section { color:#536078 !important; font-size:12px !important; font-weight:600 !important; }
    .stButton > button, .stDownloadButton > button {
      border:0 !important; border-radius:12px !important; min-height:43px;
      padding:0 20px !important; font-size:12px !important; font-weight:700 !important;
      color:white !important; background:linear-gradient(135deg,#667df7,#4d63e8) !important;
      box-shadow:0 7px 17px rgba(82,109,245,.22) !important;
      transition:transform .18s ease, box-shadow .18s ease !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
      transform:translateY(-1px); box-shadow:0 10px 22px rgba(82,109,245,.28) !important;
    }
    .stButton > button:active { transform:translateY(0); }
    [data-testid="stMetric"] {
      background:rgba(255,255,255,.64); border:1px solid rgba(255,255,255,.94);
      border-radius:17px; padding:15px 17px; box-shadow:0 7px 22px rgba(48,63,105,.045);
    }
    [data-testid="stMetricLabel"] { color:#8490a4 !important; font-size:11px !important; }
    [data-testid="stMetricValue"] { color:#253149 !important; font:750 25px 'Manrope',sans-serif !important; }
    [data-testid="stDataFrame"] {
      border:1px solid rgba(255,255,255,.95); border-radius:15px; overflow:hidden;
      background:rgba(255,255,255,.6);
    }
    [data-testid="stAlert"] { border-radius:13px; }
    .result-card {
      padding:22px 24px; border-radius:20px; margin:12px 0 17px;
      background:rgba(255,255,255,.72); border:1px solid rgba(255,255,255,.96);
      box-shadow:0 9px 26px rgba(48,63,105,.06);
    }
    .result-top { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .result-label { font:800 24px 'Manrope',sans-serif; letter-spacing:-.035em; }
    .result-score { font:700 13px 'Manrope',sans-serif; color:#65718a; }
    .result-note { color:#8792a5; font-size:11px; margin-top:6px; }
    .status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }
    .score-track { height:7px; background:#e9edf5; border-radius:20px; overflow:hidden; margin-top:17px; }
    .score-fill { height:100%; border-radius:20px; }
    .mini-heading { font:700 13px 'Manrope',sans-serif; color:#35415a; margin:8px 0 12px; }
    .footer {
      text-align:center; color:#a0a9b8; font-size:10px; letter-spacing:.04em; padding:24px 0 0;
    }
    @media(max-width:700px) {
      .block-container { padding:1.2rem 1rem 3rem; }
      .hero { padding:24px 21px; border-radius:22px; }
      .glass-card { padding:18px 16px; }
      .result-top { align-items:flex-start; flex-direction:column; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- NLP engine -----------------------------
@st.cache_resource(show_spinner=False)
def load_analyzer():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception as exc:
            raise RuntimeError("Could not download the VADER lexicon. Check your network connection.") from exc
    try:
        return SentimentIntensityAnalyzer()
    except LookupError as exc:
        raise RuntimeError("VADER lexicon is unavailable. Restart the app after checking the network.") from exc


def preprocess_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "", str(text))
    return re.sub(r"\s+", " ", text).strip()


def get_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_sentiment(analyzer, text: str) -> Tuple[str, Dict[str, float]]:
    scores = analyzer.polarity_scores(preprocess_text(text))
    return get_label(scores["compound"]), scores


def icon(name: str, size: int = 18, color: str = "currentColor") -> str:
    paths = {
        "sparkles": '<path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z"/><path d="m19 14 1.1 2.9L23 18l-2.9 1.1L19 22l-1.1-2.9L15 18l2.9-1.1L19 14Z"/>',
        "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/>',
        "chart": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-5 5"/>',
        "shield": '<path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z"/><path d="m9 12 2 2 4-4"/>',
        "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h8"/>',
        "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5M12 15V3"/>',
        "message": '<path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5 9 9 0 0 1-4-.9L3 21l1.9-5.5a9 9 0 0 1-.9-4A8.5 8.5 0 0 1 12.5 3h.5a8.5 8.5 0 0 1 8 8v.5Z"/>',
        "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    }
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths.get(name, paths["sparkles"])}</svg>'


def result_panel(label: str, compound: float):
    colors = {"Positive": "#159b72", "Negative": "#df5b68", "Neutral": "#b98228"}
    color = colors[label]
    width = max(0, min(100, (compound + 1) * 50))
    st.markdown(
        f"""
        <div class="result-card">
          <div class="section-kicker">Analysis result</div>
          <div class="result-top">
            <div class="result-label"><span class="status-dot" style="background:{color}"></span>{label} sentiment</div>
            <div class="result-score">Compound score&nbsp;&nbsp; {compound:+.3f}</div>
          </div>
          <div class="result-note">VADER compound score, normalized from −1 (most negative) to +1 (most positive).</div>
          <div class="score-track"><div class="score-fill" style="width:{width}%;background:{color}"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sentiment_rows(analyzer, reviews: List[str]) -> pd.DataFrame:
    rows = []
    for review in reviews:
        label, scores = analyze_sentiment(analyzer, review)
        rows.append({
            "Review": review,
            "Sentiment": label,
            "Compound": round(scores["compound"], 4),
            "Positive": round(scores["pos"], 4),
            "Neutral": round(scores["neu"], 4),
            "Negative": round(scores["neg"], 4),
        })
    return pd.DataFrame(rows)


# ----------------------------- Header -----------------------------
st.markdown(
    f"""
    <div class="hero">
      <div class="brand-row">
        <div class="brand-mark">{icon("activity", 20, "#ffffff")}</div>
        <div class="brand-name">SENTIMENT STUDIO</div>
      </div>
      <div class="eyebrow">Understand the tone behind the text</div>
      <h1>Sentiment, made clear.</h1>
      <p>Turn written feedback into useful signals. Analyze individual reviews or process a full batch with a clean, readable sentiment breakdown.</p>
      <div class="hero-meta">
        <span class="meta-chip">{icon("scan", 13, "#6678db")} VADER analysis</span>
        <span class="meta-chip">{icon("shield", 13, "#159b72")} Runs locally in this app</span>
        <span class="meta-chip">{icon("chart", 13, "#6678db")} Single & batch insights</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    analyzer = load_analyzer()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

tab_single, tab_batch = st.tabs(["  Single analysis  ", "  Batch analysis  "])

# ----------------------------- Single analysis -----------------------------
with tab_single:
    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="glass-card">
              <div class="section-kicker">01 / Input</div>
              <div class="section-title">{icon("message", 18, "#6479ed")} &nbsp;Analyze a review</div>
              <p class="section-copy">Paste a review, comment, or any short piece of text.</p>
            """,
            unsafe_allow_html=True,
        )
        review = st.text_area(
            "Your text",
            height=190,
            placeholder="The experience was excellent. The staff were helpful and everything arrived on time...",
            label_visibility="collapsed",
            key="single_review",
        )
        count_col, hint_col = st.columns([1, 2])
        count_col.caption(f"{len(review)} characters")
        hint_col.caption("Tip: punctuation and capitalization can influence VADER scores.")
        analyze_clicked = st.button("Analyze sentiment  →", type="primary", use_container_width=True, key="single_analyze")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="glass-card">
              <div class="section-kicker">02 / Output</div>
              <div class="section-title">Your insight</div>
              <p class="section-copy">Your classification and polarity distribution appear here.</p>
            """,
            unsafe_allow_html=True,
        )
        if analyze_clicked:
            if not review.strip():
                st.warning("Enter some text before running the analysis.")
            else:
                label, scores = analyze_sentiment(analyzer, review)
                st.session_state["single_result"] = {"label": label, "scores": scores, "review": review}
        result = st.session_state.get("single_result")
        if result:
            result_panel(result["label"], result["scores"]["compound"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Positive", f'{result["scores"]["pos"] * 100:.1f}%')
            c2.metric("Neutral", f'{result["scores"]["neu"] * 100:.1f}%')
            c3.metric("Negative", f'{result["scores"]["neg"] * 100:.1f}%')
            polarity = pd.DataFrame({
                "Score": [result["scores"]["pos"], result["scores"]["neu"], result["scores"]["neg"]],
            }, index=["Positive", "Neutral", "Negative"])
            st.markdown('<div class="mini-heading">Polarity distribution</div>', unsafe_allow_html=True)
            st.bar_chart(polarity, height=150, color="#7186f5")
        else:
            st.markdown(
                f'<div style="padding:36px 12px;text-align:center;color:#9aa4b5">{icon("chart", 30, "#aeb8ca")}<div style="font-size:12px;margin-top:12px">Your analysis will appear here.</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- Batch analysis -----------------------------
with tab_batch:
    st.markdown(
        f"""
        <div class="glass-card">
          <div class="section-kicker">01 / Batch workspace</div>
          <div class="section-title">{icon("file", 18, "#6479ed")} &nbsp;Analyze multiple reviews</div>
          <p class="section-copy">Paste one review per line, or upload a CSV file containing a review column.</p>
        """,
        unsafe_allow_html=True,
    )
    batch_text = st.text_area(
        "Reviews",
        height=180,
        placeholder="The food was delicious and the service was great.\nThe delivery was late and support did not respond.\nThe product arrived yesterday.",
        label_visibility="collapsed",
        key="batch_text",
    )
    uploaded = st.file_uploader("Or upload a CSV file", type=["csv"], help="The first text column will be used if no review/text/feedback column is found.")
    uploaded_reviews = []
    if uploaded is not None:
        try:
            csv_df = pd.read_csv(uploaded)
            preferred = next((c for c in csv_df.columns if c.strip().lower() in {"review", "text", "feedback", "comment"}), None)
            selected_column = st.selectbox("Select review column", options=list(csv_df.columns), index=list(csv_df.columns).index(preferred) if preferred else 0)
            uploaded_reviews = csv_df[selected_column].dropna().astype(str).tolist()
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    run_batch = st.button("Analyze all reviews  →", type="primary", key="batch_analyze")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_batch:
        pasted_reviews = [line.strip() for line in batch_text.splitlines() if line.strip()]
        reviews = uploaded_reviews if uploaded_reviews else pasted_reviews
        if not reviews:
            st.warning("Add at least one review or upload a CSV file.")
        else:
            st.session_state["batch_results"] = sentiment_rows(analyzer, reviews)

    batch_result = st.session_state.get("batch_results")
    if batch_result is not None and not batch_result.empty:
        st.markdown(
            """
            <div class="glass-card">
              <div class="section-kicker">02 / Results</div>
              <div class="section-title">Sentiment overview</div>
              <p class="section-copy">A row-by-row classification with VADER compound scores.</p>
            """,
            unsafe_allow_html=True,
        )
        total = len(batch_result)
        counts = batch_result["Sentiment"].value_counts()
        positive = int(counts.get("Positive", 0))
        neutral = int(counts.get("Neutral", 0))
        negative = int(counts.get("Negative", 0))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total reviews", total)
        m2.metric("Positive", positive, f"{positive / total * 100:.1f}%")
        m3.metric("Neutral", neutral, f"{neutral / total * 100:.1f}%")
        m4.metric("Negative", negative, f"{negative / total * 100:.1f}%")
        chart_col, table_col = st.columns([0.75, 1.25], gap="large")
        with chart_col:
            st.markdown('<div class="mini-heading">Classification mix</div>', unsafe_allow_html=True)
            distribution = pd.DataFrame({"Reviews": [positive, neutral, negative]}, index=["Positive", "Neutral", "Negative"])
            st.bar_chart(distribution, height=230, color="#7186f5")
        with table_col:
            st.markdown('<div class="mini-heading">Review-level results</div>', unsafe_allow_html=True)
            st.dataframe(batch_result, use_container_width=True, hide_index=True, height=270)
        csv_bytes = batch_result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="sentiment-analysis-results.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f'<div class="footer">SENTIMENT STUDIO &nbsp;·&nbsp; POWERED BY NLTK VADER &nbsp;·&nbsp; {icon("shield", 11, "#a0a9b8")} PRIVATE BY DESIGN</div>',
    unsafe_allow_html=True,
)
