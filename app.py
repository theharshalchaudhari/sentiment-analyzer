"""
Sentiment Studio — streamlined, glassmorphism sentiment analysis.
Run: streamlit run app.py
"""

import re
from typing import Dict, List, Tuple

import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(page_title="Sentiment Studio", page_icon=" ", layout="wide", initial_sidebar_state="collapsed")

# Keep this stylesheet mounted across app states. The fixed animated backdrop is
# independent of the page content, so navigation does not reset its position.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#172033;--muted:#77839a;--blue:#596ff0;--green:#1b9b76;--red:#df6570;--amber:#bd8a38}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:transparent;color:var(--ink)}
[data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#f5f7fd,#eef2fa 52%,#f8f6fc);position:relative;isolation:isolate}
[data-testid="stAppViewContainer"]::before,[data-testid="stAppViewContainer"]::after{content:"";position:fixed;z-index:0;pointer-events:none;border-radius:50%;filter:blur(75px);opacity:.58;will-change:transform}
[data-testid="stAppViewContainer"]::before{width:440px;height:440px;left:-130px;top:-130px;background:radial-gradient(circle,rgba(116,145,255,.62),rgba(116,145,255,0) 70%);animation:aurora-one 22s ease-in-out infinite alternate}
[data-testid="stAppViewContainer"]::after{width:520px;height:520px;right:-170px;top:10%;background:radial-gradient(circle,rgba(191,160,255,.52),rgba(191,160,255,0) 70%);animation:aurora-two 27s ease-in-out infinite alternate}
@keyframes aurora-one{0%{transform:translate3d(0,0,0) scale(1)}50%{transform:translate3d(30vw,18vh,0) scale(1.18)}100%{transform:translate3d(12vw,55vh,0) scale(.92)}}
@keyframes aurora-two{0%{transform:translate3d(0,0,0) scale(1)}50%{transform:translate3d(-24vw,35vh,0) scale(.9)}100%{transform:translate3d(-8vw,-12vh,0) scale(1.15)}}
@media(prefers-reduced-motion:reduce){[data-testid="stAppViewContainer"]::before,[data-testid="stAppViewContainer"]::after{animation:none}}
[data-testid="stMain"]{position:relative;z-index:1;background:transparent}[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"],#MainMenu,footer{visibility:hidden}
.block-container{max-width:1120px;padding:1.55rem 2rem 3rem}
.brandbar{display:flex;align-items:center;justify-content:space-between;margin:0 0 26px}
.brand{display:flex;align-items:center;gap:10px;color:#25314a;font:800 13px Manrope,sans-serif;letter-spacing:.08em}
.brand-icon{width:34px;height:34px;border-radius:12px;display:grid;place-items:center;color:#fff;background:linear-gradient(145deg,#8294ff,#5369eb);box-shadow:0 7px 18px #6279ef45}
.brand-status{font-size:10px;color:#7c879c;padding:7px 11px;border:1px solid #ffffffc9;background:#ffffff77;border-radius:30px}
.hero{position:relative;overflow:hidden;padding:44px 46px 40px;border-radius:30px;background:linear-gradient(120deg,#ffffffd9,#ffffff8c);border:1px solid #ffffffed;box-shadow:0 20px 60px #34436d14,inset 0 1px #fff;backdrop-filter:blur(24px);margin-bottom:20px}
.hero:after{content:"";position:absolute;width:320px;height:320px;right:-90px;top:-160px;border-radius:50%;background:radial-gradient(circle,#8da2ff44,transparent 68%);pointer-events:none}
.eyebrow{font-size:10px;font-weight:700;letter-spacing:.17em;text-transform:uppercase;color:#6579e4;margin-bottom:12px}
.hero h1{font:800 clamp(34px,5vw,54px)/1.06 Manrope,sans-serif;letter-spacing:-.055em;color:#1b263d;margin:0 0 14px}
.hero p{max-width:570px;color:#738097;font-size:14px;line-height:1.75;margin:0}
.hero-tags{display:flex;gap:8px;flex-wrap:wrap;margin-top:23px}
.tag{font-size:10px;font-weight:600;color:#68758d;padding:8px 11px;border:1px solid #fff;background:#ffffffa8;border-radius:30px}
.glass{padding:23px 24px;border:1px solid #ffffffed;border-radius:23px;background:linear-gradient(145deg,#ffffffd9,#ffffff91);box-shadow:0 13px 38px #34436d0e,inset 0 1px #fff;backdrop-filter:blur(22px);margin-bottom:15px}
.glass h3{font:750 18px Manrope,sans-serif;letter-spacing:-.025em;color:#26324a;margin:0 0 5px}
.muted{font-size:12px;color:#8390a5;line-height:1.6}
.kicker{font-size:10px;letter-spacing:.15em;text-transform:uppercase;font-weight:700;color:#8a96aa;margin-bottom:8px}
.step-number{font:800 11px Manrope,sans-serif;color:#6579e5;letter-spacing:.1em;margin-bottom:10px}
.feature{height:100%;padding:18px;border-radius:18px;background:#ffffff83;border:1px solid #ffffffdf}
.feature-icon{width:37px;height:37px;display:grid;place-items:center;border-radius:13px;background:#edf0ff;color:#6378e8;margin-bottom:13px}
.feature-title{font:700 13px Manrope,sans-serif;color:#33405a;margin-bottom:5px}
.feature-copy{font-size:11px;line-height:1.55;color:#8490a4}
.stButton>button,.stDownloadButton>button{min-height:43px;border:0!important;border-radius:12px!important;padding:0 18px!important;font-size:12px!important;font-weight:700!important;color:#fff!important;background:linear-gradient(135deg,#7185f7,#5369e8)!important;box-shadow:0 8px 20px #5a70ed38!important;transition:transform .18s ease,box-shadow .18s ease!important}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-1px);box-shadow:0 11px 24px #5a70ed48!important}
.stButton>button[kind="secondary"]{background:#ffffffb8!important;color:#586784!important;border:1px solid #ffffff!important;box-shadow:0 5px 16px #34436d0c!important}
.stTextArea textarea{background:#ffffffbd!important;border:1px solid #dce2f0!important;border-radius:14px!important;color:#293650!important;font-size:13px!important;line-height:1.65!important}
.stTextArea textarea:focus{border-color:#8293f5!important;box-shadow:0 0 0 3px #657cf01c!important}
label{font-size:12px!important;color:#596780!important;font-weight:600!important}
[data-testid="stMetric"]{padding:14px 16px;background:#ffffffa6;border:1px solid #ffffff;border-radius:16px;box-shadow:0 7px 20px #34436d0a}
[data-testid="stMetricLabel"]{font-size:10px!important;color:#8490a4!important}
[data-testid="stMetricValue"]{font:750 23px Manrope,sans-serif!important;color:#26334b!important}
[data-testid="stDataFrame"]{border:1px solid #ffffff;border-radius:15px;overflow:hidden;background:#ffffffa8}
[data-testid="stFileUploader"] section{background:#ffffff80;border:1px dashed #cbd3e6;border-radius:14px}
.result-pill{display:inline-flex;align-items:center;gap:7px;border-radius:30px;padding:7px 11px;font-size:10px;font-weight:700;background:#fff;border:1px solid #fff}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block}
.score{font:800 30px Manrope,sans-serif;letter-spacing:-.04em;color:#253149}
.footer{text-align:center;color:#9aa5b7;font-size:10px;letter-spacing:.08em;padding:22px 0 0}
hr{border-color:#ffffffa8!important}
@media(max-width:700px){.block-container{padding:1rem 1rem 2.5rem}.hero{padding:28px 23px;border-radius:23px}.glass{padding:18px}.brandbar{margin-bottom:18px}}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Helpers -----------------------------
@st.cache_resource(show_spinner=False)
def load_analyzer():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        if not nltk.download("vader_lexicon", quiet=True):
            raise RuntimeError("Could not download the VADER lexicon. Check your connection and retry.")
    return SentimentIntensityAnalyzer()

def preprocess_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "", str(text))
    return re.sub(r"\s+", " ", text).strip()

def get_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"

def analyze(analyzer, text: str) -> Tuple[str, Dict[str, float]]:
    scores = analyzer.polarity_scores(preprocess_text(text))
    return get_label(scores["compound"]), scores

def analyze_many(analyzer, reviews: List[str]) -> pd.DataFrame:
    rows = []
    for review in reviews:
        label, score = analyze(analyzer, review)
        rows.append({"Review": review, "Sentiment": label, "Compound": score["compound"],
                     "Positive": score["pos"], "Neutral": score["neu"], "Negative": score["neg"]})
    return pd.DataFrame(rows)

def donut_chart(values: Dict[str, float], center_label: str, center_value: str, height: int = 310):
    labels = ["Positive", "Neutral", "Negative"]
    colors = ["#27A982", "#E8B45D", "#E16D79"]
    fig = go.Figure(go.Pie(
        labels=labels, values=[values.get(x, 0) for x in labels], hole=.70,
        sort=False, direction="clockwise", rotation=90,
        marker=dict(colors=colors, line=dict(color="rgba(255,255,255,.95)", width=5)),
        textinfo="none", hovertemplate="<b>%{label}</b><br>%{value:.1%}<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=8,r=8,t=8,b=8), showlegend=True,
        legend=dict(orientation="h", y=-.06, x=.5, xanchor="center",
                    font=dict(family="DM Sans", size=11, color="#77839A")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"<b>{center_value}</b><br><span style='font-size:10px;color:#8792A5'>{center_label}</span>",
                          x=.5,y=.51,showarrow=False,font=dict(family="Manrope",size=23,color="#26334B"))],
        font=dict(family="DM Sans",color="#77839A"),
    )
    return fig

def brandbar():
    st.markdown("""
    <div class="brandbar">
      <div class="brand"><div class="brand-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg></div>SENTIMENT STUDIO</div>
      <div class="brand-status">VADER · TEXT ANALYTICS</div>
    </div>
    """, unsafe_allow_html=True)

def go(page: str):
    st.session_state.page = page

def back_button(target: str, label: str = "← Back"):
    if st.button(label, key=f"back_{target}", type="secondary"):
        go(target)
        st.rerun()

# ----------------------------- State -----------------------------
for key, default in {
    "page": "home", "mode": None, "single_text": "", "batch_text": "",
    "single_result": None, "batch_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

try:
    analyzer = load_analyzer()
except Exception as exc:
    brandbar()
    st.error(f"Sentiment engine could not start: {exc}")
    st.stop()

brandbar()

# ----------------------------- Landing -----------------------------
if st.session_state.page == "home":
    st.markdown("""
    <section class="hero">
      <div class="eyebrow">A clearer way to understand feedback</div>
      <h1>Read between<br>the sentiments.</h1>
      <p>Turn reviews, comments and customer feedback into clear sentiment insights. Start with one piece of text or explore patterns across an entire batch.</p>
      <div class="hero-tags"><span class="tag">VADER-powered analysis</span><span class="tag">Instant results</span><span class="tag">CSV export</span></div>
    </section>
    """, unsafe_allow_html=True)
    a,b = st.columns([1.12,.88], gap="large")
    with a:
        st.markdown("""
        <div class="glass">
          <div class="kicker">YOUR WORKSPACE</div>
          <h3>Make every response count.</h3>
          <p class="muted">Classify sentiment, understand polarity and get a visual summary without navigating a complicated dashboard.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start analyzing  →", type="primary", use_container_width=True, key="start"):
            go("choose")
            st.rerun()
    with b:
        st.markdown("""
        <div class="glass">
          <div class="kicker">BUILT FOR CLARITY</div>
          <div class="feature" style="margin-bottom:10px"><div class="feature-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6378e8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg></div><div class="feature-title">Simple input</div><div class="feature-copy">Analyze a single review or a complete list.</div></div>
          <div class="feature"><div class="feature-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6378e8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-5 5"/></svg></div><div class="feature-title">Visual insights</div><div class="feature-copy">See sentiment proportions in a clear circular chart.</div></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">SENTIMENT STUDIO · POWERED BY NLTK VADER</div>
    """, unsafe_allow_html=True)

# ----------------------------- Choose workflow -----------------------------
elif st.session_state.page == "choose":
    back_button("home", "← Home")
    st.markdown('<div class="hero"><div class="eyebrow">STEP 01 · CHOOSE YOUR WORKFLOW</div><h1>What would you<br>like to analyze?</h1><p>Choose the input that matches your task. You can change this choice at any time.</p></div>', unsafe_allow_html=True)
    one,many = st.columns(2, gap="large")
    with one:
        st.markdown("""
        <div class="glass" style="min-height:205px">
          <div class="feature-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6378e8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5 9 9 0 0 1-4-.9L3 21l1.9-5.5a9 9 0 0 1-.9-4A8.5 8.5 0 0 1 12.5 3h.5a8.5 8.5 0 0 1 8 8v.5Z"/></svg></div>
          <h3>Single text</h3><p class="muted">Analyze one review, message, comment or customer response.</p>
          <div class="kicker">ONE INPUT · INSTANT BREAKDOWN</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Choose single text  →", use_container_width=True, key="choose_single"):
            st.session_state.mode = "single"
            go("input")
            st.rerun()
    with many:
        st.markdown("""
        <div class="glass" style="min-height:205px">
          <div class="feature-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6378e8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h8"/></svg></div>
          <h3>Multiple texts</h3><p class="muted">Analyze many reviews together or import a CSV file.</p>
          <div class="kicker">BATCH INPUT · EXPORTABLE RESULTS</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Choose multiple texts  →", use_container_width=True, key="choose_many"):
            st.session_state.mode = "batch"
            go("input")
            st.rerun()

# ----------------------------- Input -----------------------------
elif st.session_state.page == "input":
    back_button("choose", "← Change analysis type")
    if st.session_state.mode == "single":
        st.markdown('<div class="hero"><div class="eyebrow">STEP 02 · SINGLE ANALYSIS</div><h1>Paste your text.</h1><p>We will identify the overall sentiment and show the positive, neutral and negative polarity distribution.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="glass"><div class="kicker">TEXT INPUT</div><h3>Your review or message</h3><p class="muted">Keep the original punctuation and capitalization for VADER analysis.</p></div>', unsafe_allow_html=True)
        text_value = st.text_area("Text to analyze", value=st.session_state.single_text, height=170, placeholder="Write or paste your review here…", label_visibility="collapsed", key="single_input")
        st.caption(f"{len(text_value)} characters")
        if st.button("Analyze sentiment  →", type="primary", use_container_width=True, key="analyze_single"):
            if not text_value.strip():
                st.warning("Enter some text to continue.")
            else:
                label, scores = analyze(analyzer, text_value)
                st.session_state.single_text = text_value
                st.session_state.single_result = {"label": label, "scores": scores, "text": text_value}
                go("results")
                st.rerun()
    else:
        st.markdown('<div class="hero"><div class="eyebrow">STEP 02 · BATCH ANALYSIS</div><h1>Bring your feedback.</h1><p>Paste one review per line or upload a CSV. Select the column containing your review text.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="glass"><div class="kicker">BATCH INPUT</div><h3>Reviews and feedback</h3><p class="muted">Use one review per line. CSV upload is optional.</p></div>', unsafe_allow_html=True)
        batch_value = st.text_area("Reviews", value=st.session_state.batch_text, height=150, placeholder="The service was excellent.\\nDelivery was late and support did not respond.\\nThe product is okay.", label_visibility="collapsed", key="batch_input")
        uploaded = st.file_uploader("Or upload a CSV file", type=["csv"], help="A review, text, feedback or comment column is detected automatically.")
        selected_reviews = []
        if uploaded is not None:
            try:
                source_df = pd.read_csv(uploaded)
                if source_df.empty or len(source_df.columns) == 0:
                    st.warning("The uploaded CSV has no rows or columns.")
                else:
                    preferred = next((c for c in source_df.columns if str(c).strip().lower() in {"review","text","feedback","comment"}), source_df.columns[0])
                    selected_col = st.selectbox("Review text column", list(source_df.columns), index=list(source_df.columns).index(preferred))
                    selected_reviews = source_df[selected_col].dropna().astype(str).tolist()
            except Exception as exc:
                st.error(f"Unable to read this CSV: {exc}")
        if st.button("Analyze all reviews  →", type="primary", use_container_width=True, key="analyze_batch"):
            pasted = [line.strip() for line in batch_value.splitlines() if line.strip()]
            reviews = selected_reviews if uploaded is not None else pasted
            if not reviews:
                st.warning("Add at least one review or upload a CSV with review text.")
            else:
                st.session_state.batch_text = batch_value
                st.session_state.batch_result = analyze_many(analyzer, reviews)
                go("results")
                st.rerun()

# ----------------------------- Results -----------------------------
elif st.session_state.page == "results":
    back_button("input", "← Edit input")
    if st.session_state.mode == "single" and st.session_state.single_result:
        result = st.session_state.single_result
        scores = result["scores"]
        st.markdown('<div class="hero" style="padding:27px 32px"><div class="eyebrow">ANALYSIS COMPLETE</div><h1>Your sentiment overview.</h1><p>A visual breakdown of the text you submitted.</p></div>', unsafe_allow_html=True)
        left,right = st.columns([.92,1.08], gap="large")
        with left:
            st.markdown('<div class="glass"><div class="kicker">OVERALL SENTIMENT</div>', unsafe_allow_html=True)
            color = {"Positive":"#1b9b76","Neutral":"#bd8a38","Negative":"#df6570"}[result["label"]]
            st.markdown(f'<div style="font:800 29px Manrope;color:{color};margin:7px 0">{result["label"]}</div><div class="muted">Compound score</div><div class="score">{scores["compound"]:+.3f}</div><div class="muted">Scale: −1.0 (negative) to +1.0 (positive)</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="glass"><div class="kicker">ANALYZED TEXT</div>', unsafe_allow_html=True)
            st.write(result["text"])
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="glass"><div class="kicker">POLARITY DISTRIBUTION</div><h3>Sentiment composition</h3><p class="muted">Share of positive, neutral and negative language signals.</p>', unsafe_allow_html=True)
            st.plotly_chart(donut_chart({"Positive":scores["pos"],"Neutral":scores["neu"],"Negative":scores["neg"]},"POLARITY","100%",300), use_container_width=True, config={"displayModeBar":False})
            st.markdown('</div>', unsafe_allow_html=True)
        p,n,ng = st.columns(3)
        p.metric("Positive", f'{scores["pos"]*100:.1f}%')
        n.metric("Neutral", f'{scores["neu"]*100:.1f}%')
        ng.metric("Negative", f'{scores["neg"]*100:.1f}%')
    elif st.session_state.mode == "batch" and st.session_state.batch_result is not None:
        df = st.session_state.batch_result
        counts = df["Sentiment"].value_counts()
        total = len(df)
        summary = {"Positive":int(counts.get("Positive",0)),"Neutral":int(counts.get("Neutral",0)),"Negative":int(counts.get("Negative",0))}
        st.markdown('<div class="hero" style="padding:27px 32px"><div class="eyebrow">ANALYSIS COMPLETE</div><h1>Your feedback, at a glance.</h1><p>Explore the sentiment mix and inspect each individual review.</p></div>', unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total reviews",total)
        m2.metric("Positive",summary["Positive"],f'{summary["Positive"]/total*100:.1f}%')
        m3.metric("Neutral",summary["Neutral"],f'{summary["Neutral"]/total*100:.1f}%')
        m4.metric("Negative",summary["Negative"],f'{summary["Negative"]/total*100:.1f}%')
        chart_col, table_col = st.columns([.8,1.2], gap="large")
        with chart_col:
            st.markdown('<div class="glass"><div class="kicker">POLARITY DISTRIBUTION</div><h3>Sentiment composition</h3>', unsafe_allow_html=True)
            st.plotly_chart(donut_chart(summary,"TOTAL",str(total),300),use_container_width=True,config={"displayModeBar":False})
            st.markdown('</div>', unsafe_allow_html=True)
        with table_col:
            st.markdown('<div class="glass"><div class="kicker">REVIEW EXPLORER</div><h3>Individual results</h3><p class="muted">Filter and inspect your analyzed feedback.</p>', unsafe_allow_html=True)
            filter_value = st.selectbox("Filter sentiment",["All","Positive","Neutral","Negative"],label_visibility="collapsed")
            view = df if filter_value == "All" else df[df["Sentiment"] == filter_value]
            st.dataframe(view, use_container_width=True, hide_index=True, height=285)
            st.markdown('</div>', unsafe_allow_html=True)
        st.download_button("Download results as CSV",data=df.to_csv(index=False).encode("utf-8-sig"),file_name="sentiment-analysis-results.csv",mime="text/csv",use_container_width=True)
    else:
        st.info("No analysis result is available. Start a new analysis.")
        if st.button("Back to home"):
            go("home")
            st.rerun()

st.markdown('<div class="footer">SENTIMENT STUDIO · POWERED BY NLTK VADER</div>', unsafe_allow_html=True)
