"""
Terna Sentiment Analyzer — streamlined, glassmorphism sentiment analysis.
Run: streamlit run app.py
"""

import re
from typing import Dict, List, Tuple

import nltk
import pandas as pd
import plotly.graph_objects as plotly_go
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(page_title="Terna Sentiment Analyzer", page_icon=" ", layout="wide", initial_sidebar_state="collapsed")

# Keep this stylesheet mounted across app states. The fixed animated backdrop is
# independent of the page content, so navigation does not reset its position.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400&display=swap');
:root{--ink:#172033;--muted:#77839a;--blue:#596ff0;--green:#1b9b76;--red:#df6570;--amber:#bd8a38}
html,body,[class*="css"]{font-family:'Poppins',sans-serif}
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
.brand{display:flex;align-items:center;gap:10px;color:#25314a;font:800 13px Poppins,sans-serif;letter-spacing:.08em}
.brand-icon{width:34px;height:34px;border-radius:12px;display:grid;place-items:center;color:#fff;background:linear-gradient(145deg,#8294ff,#5369eb);box-shadow:0 7px 18px #6279ef45}
.brand-status{font-size:10px;color:#7c879c;padding:7px 11px;border:1px solid #ffffffc9;background:#ffffff77;border-radius:30px}
.hero{position:relative;overflow:hidden;padding:44px 46px 40px;border-radius:30px;background:linear-gradient(120deg,#ffffffd9,#ffffff8c);border:1px solid #ffffffed;box-shadow:0 20px 60px #34436d14,inset 0 1px #fff;backdrop-filter:blur(24px);margin-bottom:20px}
.hero:after{content:"";position:absolute;width:320px;height:320px;right:-90px;top:-160px;border-radius:50%;background:radial-gradient(circle,#8da2ff44,transparent 68%);pointer-events:none}
.eyebrow{font-size:10px;font-weight:700;letter-spacing:.17em;text-transform:uppercase;color:#6579e4;margin-bottom:12px}
.hero h1{font:800 clamp(34px,5vw,54px)/1.06 Poppins,sans-serif;letter-spacing:-.055em;color:#1b263d;margin:0 0 14px}
.hero p{max-width:570px;color:#738097;font-size:14px;line-height:1.75;margin:0}
.hero-tags{display:flex;gap:8px;flex-wrap:wrap;margin-top:23px}
.tag{font-size:10px;font-weight:600;color:#68758d;padding:8px 11px;border:1px solid #fff;background:#ffffffa8;border-radius:30px}
.glass{padding:23px 24px;border:1px solid #ffffffed;border-radius:23px;background:linear-gradient(145deg,#ffffffd9,#ffffff91);box-shadow:0 13px 38px #34436d0e,inset 0 1px #fff;backdrop-filter:blur(22px);margin-bottom:15px}
.glass h3{font:750 18px Poppins,sans-serif;letter-spacing:-.025em;color:#26324a;margin:0 0 5px}
.muted{font-size:12px;color:#8390a5;line-height:1.6}
.kicker{font-size:10px;letter-spacing:.15em;text-transform:uppercase;font-weight:700;color:#8a96aa;margin-bottom:8px}
.step-number{font:800 11px Poppins,sans-serif;color:#6579e5;letter-spacing:.1em;margin-bottom:10px}
.feature{height:100%;padding:18px;border-radius:18px;background:#ffffff83;border:1px solid #ffffffdf}
.feature-icon{width:37px;height:37px;display:grid;place-items:center;border-radius:13px;background:#edf0ff;color:#6378e8;margin-bottom:13px}
.feature-title{font:700 13px Poppins,sans-serif;color:#33405a;margin-bottom:5px}
.feature-copy{font-size:11px;line-height:1.55;color:#8490a4}
.stButton>button,.stDownloadButton>button{min-height:43px;border:0!important;border-radius:12px!important;padding:0 18px!important;font-size:12px!important;font-weight:700!important;color:#fff!important;background:linear-gradient(135deg,#7185f7,#5369e8)!important;box-shadow:0 8px 20px #5a70ed38!important;transition:transform .18s ease,box-shadow .18s ease!important}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-1px);box-shadow:0 11px 24px #5a70ed48!important}
.stButton>button[kind="secondary"]{background:#ffffffb8!important;color:#586784!important;border:1px solid #ffffff!important;box-shadow:0 5px 16px #34436d0c!important}
.stTextArea textarea{background:#ffffffbd!important;border:1px solid #dce2f0!important;border-radius:14px!important;color:#293650!important;font-size:13px!important;line-height:1.65!important}
.stTextArea textarea:focus{border-color:#8293f5!important;box-shadow:0 0 0 3px #657cf01c!important}
label{font-size:12px!important;color:#596780!important;font-weight:600!important}
[data-testid="stMetric"]{padding:14px 16px;background:#ffffffa6;border:1px solid #ffffff;border-radius:16px;box-shadow:0 7px 20px #34436d0a}
[data-testid="stMetricLabel"]{font-size:10px!important;color:#8490a4!important}
[data-testid="stMetricValue"]{font:750 23px Poppins,sans-serif!important;color:#26334b!important}
[data-testid="stDataFrame"]{border:1px solid #ffffff;border-radius:15px;overflow:hidden;background:#ffffffa8}
[data-testid="stFileUploader"] section{background:#ffffff80;border:1px dashed #cbd3e6;border-radius:14px}
.result-pill{display:inline-flex;align-items:center;gap:7px;border-radius:30px;padding:7px 11px;font-size:10px;font-weight:700;background:#fff;border:1px solid #fff}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block}
.score{font:800 30px Poppins,sans-serif;letter-spacing:-.04em;color:#253149}
.footer{text-align:center;color:#9aa5b7;font-size:10px;letter-spacing:.08em;padding:22px 0 0}
hr{border-color:#ffffffa8!important}
@media(max-width:700px){.block-container{padding:1rem 1rem 2.5rem}.hero{padding:28px 23px;border-radius:23px}.glass{padding:18px}.brandbar{margin-bottom:18px}}

/* Reference-inspired SaaS landing visual system */
[data-testid="stAppViewContainer"]{background:#fbfbff;position:relative;isolation:isolate;overflow:hidden}
[data-testid="stAppViewContainer"]::before,[data-testid="stAppViewContainer"]::after{content:"";position:fixed;z-index:0;pointer-events:none;border-radius:50%;filter:blur(95px);opacity:.7;will-change:transform}
[data-testid="stAppViewContainer"]::before{width:560px;height:560px;left:-180px;top:17%;background:radial-gradient(circle,rgba(244,166,231,.55),rgba(255,194,221,.25) 38%,transparent 72%);animation:aurora-one 25s ease-in-out infinite alternate}
[data-testid="stAppViewContainer"]::after{width:650px;height:650px;right:-210px;top:28%;background:radial-gradient(circle,rgba(139,158,255,.5),rgba(187,168,255,.24) 42%,transparent 72%);animation:aurora-two 29s ease-in-out infinite alternate}
[data-testid="stMain"]{position:relative;z-index:1;background:transparent}
.block-container{max-width:1320px;padding:1.15rem 2.3rem 3rem}
.brandbar{display:flex;align-items:center;justify-content:space-between;margin:0 auto 24px;max-width:1220px;padding:0 5px}
.brand{gap:10px;color:#202642;font:800 18px Poppins,sans-serif;letter-spacing:-.04em}
.brand-icon{width:35px;height:35px;border-radius:12px;background:linear-gradient(145deg,#4266f5,#7a35dc);box-shadow:0 7px 20px #634cf044}
.brand-status{font-size:10px;color:#777f99;padding:8px 13px;border:1px solid #e9eafa;background:#ffffffa8;border-radius:30px}
.landing-nav{display:flex;align-items:center;justify-content:space-between;gap:22px;margin:-54px 0 58px 230px;min-height:38px;color:#34394e;font-size:12px;font-weight:600}
.landing-nav-links{display:flex;align-items:center;gap:36px}
.landing-nav-right{display:flex;align-items:center;gap:25px}
.landing-hero{text-align:center;position:relative;padding:0 12px 0;min-height:440px}
.announcement{display:inline-flex;align-items:center;gap:8px;padding:9px 15px;border-radius:30px;background:#f2edff;border:1px solid #e9e0ff;color:#7554cf;font-size:11px;font-weight:700;margin-bottom:26px}
.landing-title{font:800 clamp(38px,5.1vw,70px)/1.08 Poppins,sans-serif;letter-spacing:-.065em;margin:0 auto 18px;max-width:1000px;background:linear-gradient(95deg,#122caa 5%,#3446e7 33%,#8d2cdb 63%,#e34f9a 86%,#ed7957 100%);-webkit-background-clip:text;background-clip:text;color:transparent}
.landing-subtitle{font-size:16px;line-height:1.65;color:#707991;max-width:650px;margin:0 auto 28px}
.landing-actions{display:flex;justify-content:center;gap:14px;margin:0 auto 26px}
.landing-actions .stButton{width:auto}
.landing-actions .stButton>button{min-width:190px;min-height:52px;border-radius:14px!important;background:linear-gradient(105deg,#2365ef,#642be7)!important;box-shadow:0 9px 24px #573ce83b!important}
.landing-actions .stButton>button[kind="secondary"]{background:#ffffffb8!important;border:1px solid #e8e9f2!important;color:#30364e!important;box-shadow:0 5px 18px #3a47750d!important}
.trust-row{display:flex;justify-content:center;align-items:center;gap:34px;flex-wrap:wrap;color:#68718a;font-size:11px;font-weight:600;margin:0 auto 30px}
.trust-item{display:flex;align-items:center;gap:8px}
.trust-check{display:grid;place-items:center;width:17px;height:17px;border-radius:50%;background:#ece8ff;color:#6642d8;font-size:11px;font-weight:800}
.dashboard-preview{position:relative;z-index:2;max-width:1040px;margin:0 auto 15px;border:1px solid #fff;border-radius:28px;background:#ffffffd9;box-shadow:0 24px 85px #46518a20,0 4px 20px #6a74a20d;backdrop-filter:blur(25px);overflow:hidden;text-align:left}
.preview-shell{display:grid;grid-template-columns:185px 1fr;min-height:440px}
.preview-sidebar{padding:25px 18px;border-right:1px solid #f0f1f8;background:#ffffffa8}
.preview-brand{display:flex;align-items:center;gap:8px;font:800 12px Poppins;color:#29334e;margin-bottom:28px}
.preview-mark{width:19px;height:19px;border-radius:7px;background:linear-gradient(135deg,#446af4,#8d3cdb)}
.preview-nav{padding:10px 12px;border-radius:10px;color:#7d879c;font-size:11px;margin:5px 0}
.preview-nav.active{background:#f0edff;color:#6b4bce;font-weight:700}
.preview-main{padding:25px 27px}
.preview-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:22px}
.preview-head strong{font:750 17px Poppins;color:#27334b}
.preview-head span{font-size:10px;color:#8b95a8}
.preview-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:15px}
.preview-metric{padding:14px 15px;border:1px solid #eff0f7;border-radius:14px;background:#ffffffc9}
.preview-metric small{display:block;font-size:9px;color:#8791a5;margin-bottom:9px}
.preview-metric b{display:block;font:800 22px Poppins;color:#29354e}
.preview-metric em{display:block;font-style:normal;font-size:9px;color:#24a37c;margin-top:5px}
.preview-bottom{display:grid;grid-template-columns:1.25fr .75fr;gap:13px}
.preview-panel{padding:17px;border:1px solid #eff0f7;border-radius:14px;background:#ffffffc9;min-height:190px}
.preview-panel-title{font:700 11px Poppins;color:#3a455d;margin-bottom:15px}
.mini-bars{height:112px;display:flex;align-items:end;gap:9px;padding:0 7px;border-bottom:1px solid #edf0f6;background:repeating-linear-gradient(to bottom,transparent 0,transparent 26px,#f0f1f7 27px)}
.mini-bars i{display:block;flex:1;border-radius:6px 6px 0 0;background:linear-gradient(180deg,#a78bf3,#6175ec);opacity:.82}
.sentiment-legend{display:flex;flex-direction:column;gap:13px;padding-top:8px}
.sentiment-row{display:flex;align-items:center;gap:8px;font-size:10px;color:#778198}
.sentiment-dot{width:8px;height:8px;border-radius:50%}
.sentiment-row b{margin-left:auto;color:#3b4660}
.float-tile{position:absolute;z-index:1;width:76px;height:76px;border-radius:21px;background:#ffffff80;border:1px solid #ffffffed;box-shadow:0 12px 32px #6d70b21b;backdrop-filter:blur(12px);display:grid;place-items:center;color:#604ee0}
.float-tile.left{left:4%;top:49%}.float-tile.right{right:4%;top:46%}.float-tile.far{right:0;top:80%;width:66px;height:66px}
.float-tile svg{width:28px;height:28px}
.footer{text-align:center;color:#a0a7b7;font-size:10px;letter-spacing:.08em;padding:24px 0 0}
@media(max-width:850px){.landing-nav{display:none}.block-container{padding:1rem 1rem 2.5rem}.landing-hero{min-height:390px}.preview-shell{grid-template-columns:1fr}.preview-sidebar{display:none}.preview-main{padding:18px}.float-tile{display:none}}
@media(max-width:560px){.landing-title{font-size:39px}.landing-subtitle{font-size:13px}.landing-actions{flex-direction:column;align-items:stretch;padding:0 15px}.landing-actions .stButton>button{width:100%}.trust-row{gap:12px}.preview-metrics{grid-template-columns:1fr}.preview-bottom{grid-template-columns:1fr}.preview-head strong{font-size:14px}}


/* Full-viewport, responsive landing canvas */
[data-testid="stAppViewContainer"]{
  min-height:100svh;
  background:linear-gradient(115deg,#fffaff 0%,#f5f5ff 28%,#fff5fb 52%,#f1f4ff 76%,#fffaff 100%);
  background-size:300% 300%;
  animation:canvas-gradient-drift 24s ease-in-out infinite;
}
@keyframes canvas-gradient-drift{
  0%{background-position:0% 40%}
  50%{background-position:100% 60%}
  100%{background-position:0% 40%}
}
[data-testid="stAppViewContainer"]::before{
  width:min(54vw,680px);height:min(54vw,680px);left:-12vw;top:8vh;
  background:radial-gradient(circle,rgba(248,154,222,.55) 0%,rgba(248,154,222,.24) 34%,transparent 72%);
  animation:aurora-one 17s ease-in-out infinite alternate;
}
[data-testid="stAppViewContainer"]::after{
  width:min(62vw,780px);height:min(62vw,780px);right:-14vw;top:12vh;
  background:radial-gradient(circle,rgba(133,151,255,.52) 0%,rgba(183,157,255,.23) 38%,transparent 72%);
  animation:aurora-two 21s ease-in-out infinite alternate;
}
@keyframes aurora-one{
  0%{transform:translate3d(-2vw,-3vh,0) scale(.88)}
  35%{transform:translate3d(27vw,9vh,0) scale(1.16)}
  70%{transform:translate3d(47vw,39vh,0) scale(.98)}
  100%{transform:translate3d(12vw,54vh,0) scale(1.2)}
}
@keyframes aurora-two{
  0%{transform:translate3d(0,0,0) scale(1)}
  35%{transform:translate3d(-25vw,23vh,0) scale(1.18)}
  70%{transform:translate3d(-43vw,48vh,0) scale(.9)}
  100%{transform:translate3d(-8vw,-5vh,0) scale(1.12)}
}
/* Home only: no document scrolling; content is sized to the viewport */
.block-container:has(.landing-hero){
  width:100%!important;max-width:none!important;
  height:100svh!important;min-height:100svh!important;max-height:100svh!important;
  overflow:hidden!important;box-sizing:border-box!important;
  padding:10px clamp(18px,4.2vw,68px) 12px!important;
}
.block-container:has(.landing-hero) .brandbar{
  max-width:1420px;margin:0 auto 8px!important;padding:0 3px;
  min-height:38px;
}
.block-container:has(.landing-hero) .brand{font-size:17px}
.block-container:has(.landing-hero) .landing-nav{
  max-width:1320px;margin:0 auto 12px!important;min-height:30px;
  padding:0 4px;font-size:11px;
}
.block-container:has(.landing-hero) .landing-nav-links{gap:clamp(15px,3vw,38px)}
.block-container:has(.landing-hero) .landing-nav-right{gap:20px}
.block-container:has(.landing-hero) .landing-hero{
  min-height:0!important;padding:0 10px!important;margin:0 auto 12px;
}
.block-container:has(.landing-hero) .announcement{
  margin-bottom:12px;padding:7px 13px;font-size:10px;
  background:#ffffff55;border:1px solid #ffffffa8;
  backdrop-filter:blur(28px) saturate(165%);-webkit-backdrop-filter:blur(28px) saturate(165%);
}
.block-container:has(.landing-hero) .landing-title{
  font-size:clamp(34px,4.5vw,62px);line-height:1.02;margin-bottom:10px;
}
.block-container:has(.landing-hero) .landing-subtitle{
  font-size:clamp(12px,1.15vw,15px);line-height:1.5;margin-bottom:15px;
}
.block-container:has(.landing-hero) .landing-actions{
  width:min(420px,100%);margin:0 auto 12px!important;gap:12px;
}
.block-container:has(.landing-hero) .landing-actions [data-testid="column"]{min-width:0}
.block-container:has(.landing-hero) .landing-actions .stButton>button{
  min-width:0;width:100%;min-height:43px;padding:0 12px!important;
  border:1px solid #ffffff8c!important;
  backdrop-filter:blur(30px) saturate(170%);-webkit-backdrop-filter:blur(30px) saturate(170%);
}
.block-container:has(.landing-hero) .landing-actions .stButton>button[kind="secondary"]{
  background:#ffffff55!important;color:#30364e!important;border:1px solid #ffffffb8!important;
}
.block-container:has(.landing-hero) .trust-row{
  gap:clamp(12px,2.8vw,32px);margin:0 auto 14px;font-size:10px;
}
.block-container:has(.landing-hero) .trust-check{
  background:#ffffff75;border:1px solid #ffffffc9;
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
}
.block-container:has(.landing-hero) .dashboard-preview{
  width:min(100%,1050px);max-width:1050px;margin:0 auto!important;
  border:1px solid #ffffffb8;border-radius:23px;
  background:rgba(255,255,255,.42);
  box-shadow:0 22px 80px rgba(75,82,155,.16),inset 0 1px rgba(255,255,255,.9);
  backdrop-filter:blur(36px) saturate(170%);-webkit-backdrop-filter:blur(36px) saturate(170%);
}
.block-container:has(.landing-hero) .preview-shell{min-height:0;height:clamp(230px,34vh,315px)}
.block-container:has(.landing-hero) .preview-sidebar{
  padding:18px 14px;background:rgba(255,255,255,.22);
  backdrop-filter:blur(32px);-webkit-backdrop-filter:blur(32px);
}
.block-container:has(.landing-hero) .preview-main{padding:18px 21px}
.block-container:has(.landing-hero) .preview-head{margin-bottom:12px}
.block-container:has(.landing-hero) .preview-head strong{font-size:14px}
.block-container:has(.landing-hero) .preview-metrics{gap:9px;margin-bottom:9px}
.block-container:has(.landing-hero) .preview-metric{
  padding:10px 12px;border:1px solid rgba(255,255,255,.8);
  background:rgba(255,255,255,.36);border-radius:13px;
  backdrop-filter:blur(30px) saturate(160%);-webkit-backdrop-filter:blur(30px) saturate(160%);
}
.block-container:has(.landing-hero) .preview-metric b{font-size:19px}
.block-container:has(.landing-hero) .preview-bottom{gap:9px}
.block-container:has(.landing-hero) .preview-panel{
  min-height:115px;padding:12px;border:1px solid rgba(255,255,255,.82);
  background:rgba(255,255,255,.32);border-radius:13px;
  backdrop-filter:blur(30px) saturate(160%);-webkit-backdrop-filter:blur(30px) saturate(160%);
}
.block-container:has(.landing-hero) .mini-bars{height:66px}
.block-container:has(.landing-hero) .sentiment-legend{gap:8px;padding-top:2px}
.block-container:has(.landing-hero) .float-tile{
  background:rgba(255,255,255,.27);border:1px solid rgba(255,255,255,.82);
  box-shadow:0 14px 40px rgba(87,87,160,.13),inset 0 1px rgba(255,255,255,.9);
  backdrop-filter:blur(36px) saturate(180%);-webkit-backdrop-filter:blur(36px) saturate(180%);
  animation:float-glass 7s ease-in-out infinite;
}
.block-container:has(.landing-hero) .float-tile.right{animation-delay:-2.2s}
.block-container:has(.landing-hero) .float-tile.far{animation-delay:-4.1s}
@keyframes float-glass{0%,100%{transform:translate3d(0,0,0) rotate(-4deg)}50%{transform:translate3d(0,-12px,0) rotate(4deg)}}
.block-container:has(.landing-hero) .footer{padding:7px 0 0;font-size:9px}
@media(max-height:760px) and (min-width:851px){
  .block-container:has(.landing-hero){padding-top:7px!important;padding-bottom:7px!important}
  .block-container:has(.landing-hero) .brandbar{min-height:30px;margin-bottom:4px!important}
  .block-container:has(.landing-hero) .landing-nav{margin-bottom:7px!important}
  .block-container:has(.landing-hero) .landing-title{font-size:clamp(31px,4vw,49px)}
  .block-container:has(.landing-hero) .announcement{margin-bottom:7px}
  .block-container:has(.landing-hero) .landing-subtitle{margin-bottom:9px}
  .block-container:has(.landing-hero) .trust-row{margin-bottom:8px}
  .block-container:has(.landing-hero) .preview-shell{height:clamp(190px,31vh,255px)}
  .block-container:has(.landing-hero) .preview-panel{min-height:95px}
}
@media(max-width:850px){
  .block-container:has(.landing-hero){
    height:100svh!important;min-height:100svh!important;max-height:100svh!important;
    padding:12px 18px!important;overflow:hidden!important;
  }
  .block-container:has(.landing-hero) .brandbar{margin-bottom:12px!important}
  .block-container:has(.landing-hero) .landing-nav{display:none}
  .block-container:has(.landing-hero) .landing-hero{padding-top:clamp(12px,5vh,38px)!important}
  .block-container:has(.landing-hero) .landing-title{font-size:clamp(34px,7vw,55px)}
  .block-container:has(.landing-hero) .preview-shell{height:clamp(205px,32vh,290px)}
  .block-container:has(.landing-hero) .preview-sidebar{display:none}
  .block-container:has(.landing-hero) .preview-shell{grid-template-columns:1fr}
  .block-container:has(.landing-hero) .float-tile{display:none}
}
@media(max-width:560px){
  .block-container:has(.landing-hero){padding:10px 13px!important}
  .block-container:has(.landing-hero) .brand-status{font-size:8px;padding:6px 8px}
  .block-container:has(.landing-hero) .landing-hero{padding-top:clamp(10px,4vh,28px)!important}
  .block-container:has(.landing-hero) .landing-title{font-size:clamp(32px,9vw,44px)}
  .block-container:has(.landing-hero) .landing-subtitle{font-size:12px;max-width:360px}
  .block-container:has(.landing-hero) .landing-actions{max-width:350px;gap:8px}
  .block-container:has(.landing-hero) .landing-actions .stButton>button{font-size:10px!important;min-height:40px}
  .block-container:has(.landing-hero) .trust-row{gap:8px 13px;font-size:9px}
  .block-container:has(.landing-hero) .preview-main{padding:13px}
  .block-container:has(.landing-hero) .preview-metrics{grid-template-columns:repeat(3,minmax(0,1fr))}
  .block-container:has(.landing-hero) .preview-metric{padding:8px 7px}
  .block-container:has(.landing-hero) .preview-metric b{font-size:15px}
  .block-container:has(.landing-hero) .preview-metric small{font-size:8px}
  .block-container:has(.landing-hero) .preview-metric em{font-size:7px}
  .block-container:has(.landing-hero) .preview-bottom{grid-template-columns:1fr 1fr}
  .block-container:has(.landing-hero) .preview-panel{padding:9px;min-height:92px}
  .block-container:has(.landing-hero) .preview-panel-title{font-size:9px;margin-bottom:8px}
  .block-container:has(.landing-hero) .mini-bars{height:50px;gap:4px}
  .block-container:has(.landing-hero) .sentiment-row{font-size:8px;gap:5px}
  .block-container:has(.landing-hero) .preview-head strong{font-size:12px}
  .block-container:has(.landing-hero) .preview-head span{font-size:8px}
}
@media(prefers-reduced-motion:reduce){
  [data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]::before,[data-testid="stAppViewContainer"]::after,
  .block-container:has(.landing-hero) .float-tile{animation:none!important}
}


/* Prevent the Streamlit main viewport itself from creating a second scroll area */
[data-testid="stMain"]:has(.landing-hero){
  height:100svh!important;max-height:100svh!important;overflow:hidden!important;
}
[data-testid="stMain"]:has(.landing-hero) [data-testid="stMainBlockContainer"]{
  height:100%!important;max-height:100%!important;overflow:hidden!important;
}
@media(max-height:700px) and (max-width:850px){
  .block-container:has(.landing-hero) .landing-hero{padding-top:8px!important}
  .block-container:has(.landing-hero) .landing-title{font-size:clamp(29px,7vw,40px);margin-bottom:7px}
  .block-container:has(.landing-hero) .landing-subtitle{margin-bottom:9px;line-height:1.35}
  .block-container:has(.landing-hero) .announcement{margin-bottom:8px}
  .block-container:has(.landing-hero) .trust-row{margin-bottom:8px}
  .block-container:has(.landing-hero) .preview-shell{height:clamp(150px,27vh,190px)}
  .block-container:has(.landing-hero) .preview-bottom{display:none}
  .block-container:has(.landing-hero) .preview-head{margin-bottom:7px}
}


/* Cogniscan-inspired visual language: editorial typography, clean white canvas,
   generous spacing and high-contrast black actions. */
[data-testid="stAppViewContainer"]{background:#f8f8f6!important}
[data-testid="stAppViewContainer"]::before,[data-testid="stAppViewContainer"]::after{opacity:.2!important;filter:blur(100px)!important}
.block-container{max-width:1240px;padding:1.2rem 3.2rem 2.5rem}
.brandbar{justify-content:flex-start;margin:0 0 18px}
.brand{font-size:14px;letter-spacing:.025em;color:#111827}
.brand-icon{border-radius:50%;background:#111827;box-shadow:none}
.brand-status{display:none}
.landing-hero{min-height:300px;text-align:left!important;display:flex;flex-direction:column;align-items:flex-start!important;justify-content:center!important;padding:48px 0 24px!important;margin:0!important}
.landing-hero .announcement{background:#fff!important;border:1px solid #e8e8e8!important;color:#4b5563!important;border-radius:999px!important}
.landing-title{max-width:850px!important;text-align:left!important;color:#111!important;font-family:'Poppins',sans-serif!important;font-weight:500!important;letter-spacing:-.065em!important;line-height:1.02!important;font-size:clamp(42px,6.3vw,78px)!important}
.landing-title span,.landing-title strong{color:#111!important}
.landing-subtitle{max-width:650px!important;text-align:left!important;color:#6b7280!important;font-size:clamp(15px,1.5vw,19px)!important;line-height:1.7!important;margin:18px 0 0!important}
.landing-actions{max-width:460px!important;margin:10px 0 0!important}
.landing-actions .stButton>button{background:#111!important;color:#fff!important;border-radius:999px!important;box-shadow:none!important;min-height:54px!important;font-size:14px!important}
.landing-actions .stButton>button:hover{background:#303030!important;transform:translateY(-2px)}
.landing-actions .stButton>button[kind="secondary"]{background:transparent!important;color:#111!important;border:1px solid #d1d5db!important}
.trust-row{justify-content:flex-start!important;margin:18px 0 26px!important}
.dashboard-preview{margin-top:24px!important}
.preview-shell{border-radius:22px!important;box-shadow:0 24px 70px #11111112!important;border:1px solid #e5e7eb!important}
.footer{color:#9ca3af!important}
@media(max-width:700px){
 .block-container{padding:1rem 1rem 2rem!important}
 .landing-hero{min-height:250px!important;padding:32px 0 18px!important}
 .landing-title{font-size:clamp(39px,11vw,58px)!important}
 .landing-actions{max-width:none!important}
 .trust-row{flex-wrap:wrap!important;gap:10px!important}
}


@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
/* Centered hero, one prominent CTA, and a deliberately cropped dashboard card. */
.block-container:has(.landing-hero){display:flex!important;flex-direction:column!important;height:100svh!important;min-height:100svh!important;max-height:100svh!important;overflow:hidden!important;padding:12px clamp(18px,4.5vw,72px) 0!important}
.block-container:has(.landing-hero) .brandbar{flex:0 0 auto;margin:0 auto!important;width:min(100%,1420px)}
.block-container:has(.landing-hero) .landing-hero{flex:0 0 auto;width:100%;min-height:0!important;margin:clamp(52px,11vh,118px) auto 0!important;padding:0 12px!important;text-align:center!important;display:flex;flex-direction:column;align-items:center;justify-content:center}
.block-container:has(.landing-hero) .landing-title{max-width:1080px!important;margin:0 auto 17px!important;font-family:'Poppins',sans-serif!important;font-weight:600!important;font-size:clamp(36px,5.1vw,68px)!important;line-height:1.14!important;letter-spacing:-.055em!important}
.block-container:has(.landing-hero) .landing-subtitle{max-width:740px!important;margin:0 auto!important;font-family:'Poppins',sans-serif!important;font-size:clamp(13px,1.25vw,16px)!important;line-height:1.8!important;color:#697386!important}
.block-container:has(.landing-hero) .landing-actions{flex:0 0 auto;width:min(100%,330px)!important;margin:26px auto 0!important;display:flex!important;justify-content:center!important}
.block-container:has(.landing-hero) .landing-actions .stButton{width:100%!important}
.block-container:has(.landing-hero) .landing-actions .stButton>button{width:100%!important;min-height:64px!important;padding:0 32px!important;border-radius:999px!important;border:1px solid rgba(255,255,255,.75)!important;background:#171923!important;color:#fff!important;box-shadow:0 14px 32px rgba(25,29,48,.18)!important;font-family:'Poppins',sans-serif!important;font-size:17px!important;font-weight:600!important;letter-spacing:.005em!important;transition:transform .2s ease,box-shadow .2s ease,background .2s ease!important}
.block-container:has(.landing-hero) .landing-actions .stButton>button:hover{transform:translateY(-3px)!important;background:#292d3b!important;box-shadow:0 18px 38px rgba(25,29,48,.24)!important}
.block-container:has(.landing-hero) .dashboard-preview{flex:0 0 auto;width:min(100%,1120px)!important;max-width:1120px!important;height:31svh!important;min-height:190px!important;max-height:290px!important;margin:clamp(38px,7vh,76px) auto 0!important;overflow:hidden!important;border-radius:27px 27px 0 0!important;border:1px solid rgba(255,255,255,.9)!important;background:rgba(255,255,255,.58)!important;box-shadow:0 24px 90px rgba(55,65,125,.17),0 2px 12px rgba(55,65,125,.05)!important;backdrop-filter:blur(34px) saturate(165%)!important}
.block-container:has(.landing-hero) .preview-shell{height:470px!important;min-height:470px!important;grid-template-columns:190px minmax(0,1fr)!important}
.block-container:has(.landing-hero) .preview-sidebar{padding:24px 17px!important;background:rgba(255,255,255,.52)!important;border-right:1px solid rgba(222,226,239,.8)!important}
.block-container:has(.landing-hero) .preview-brand{font-family:'Poppins',sans-serif!important;font-size:11px!important;letter-spacing:-.02em;margin-bottom:26px!important}
.block-container:has(.landing-hero) .preview-nav{border-radius:11px!important;padding:11px 12px!important;font-family:'Poppins',sans-serif!important;font-size:10px!important}
.block-container:has(.landing-hero) .preview-nav.active{background:#eef0ff!important;color:#5268db!important}
.block-container:has(.landing-hero) .preview-main{padding:25px 28px!important}
.block-container:has(.landing-hero) .preview-head{margin-bottom:19px!important}
.block-container:has(.landing-hero) .preview-head strong{font-family:'Poppins',sans-serif!important;font-size:16px!important;font-weight:600!important;color:#20283d!important}
.block-container:has(.landing-hero) .preview-head span{font-family:'Poppins',sans-serif!important;font-size:9px!important;letter-spacing:.08em}
.block-container:has(.landing-hero) .preview-metric{padding:16px!important;border-radius:16px!important;border:1px solid #e9ecf5!important;background:rgba(255,255,255,.78)!important;box-shadow:0 5px 18px rgba(40,50,90,.035)!important}
.block-container:has(.landing-hero) .preview-metric small{font-family:'Poppins',sans-serif!important;font-size:10px!important;color:#7c8598!important}
.block-container:has(.landing-hero) .preview-metric b{font-family:'Poppins',sans-serif!important;font-size:25px!important;color:#20283d!important}
.block-container:has(.landing-hero) .preview-metric em{font-family:'Poppins',sans-serif!important;font-size:9px!important}
.block-container:has(.landing-hero) .preview-panel{border-radius:16px!important;border:1px solid #e9ecf5!important;background:rgba(255,255,255,.78)!important;min-height:165px!important}
.block-container:has(.landing-hero) .preview-panel-title{font-family:'Poppins',sans-serif!important;font-size:11px!important}
.block-container:has(.landing-hero) .sentiment-row{font-family:'Poppins',sans-serif!important;font-size:10px!important}
.block-container:has(.landing-hero) .footer{display:none!important}
@media(max-width:850px){.block-container:has(.landing-hero) .landing-hero{margin-top:clamp(64px,13vh,115px)!important}.block-container:has(.landing-hero) .dashboard-preview{margin-top:clamp(34px,6vh,60px)!important;height:29svh!important}.block-container:has(.landing-hero) .preview-shell{grid-template-columns:1fr!important}.block-container:has(.landing-hero) .preview-sidebar{display:none!important}}
@media(max-width:560px){.block-container:has(.landing-hero){padding:12px 14px 0!important}.block-container:has(.landing-hero) .brand{font-size:14px!important}.block-container:has(.landing-hero) .landing-hero{margin-top:clamp(70px,15vh,125px)!important;padding:0 4px!important}.block-container:has(.landing-hero) .landing-title{font-size:clamp(34px,9.4vw,46px)!important;line-height:1.16!important}.block-container:has(.landing-hero) .landing-subtitle{font-size:12px!important;line-height:1.75!important}.block-container:has(.landing-hero) .landing-actions{width:min(100%,300px)!important;margin-top:24px!important}.block-container:has(.landing-hero) .landing-actions .stButton>button{min-height:60px!important;font-size:16px!important;padding:0 26px!important}.block-container:has(.landing-hero) .dashboard-preview{height:27svh!important;min-height:165px!important;margin-top:clamp(32px,6vh,48px)!important;border-radius:22px 22px 0 0!important}.block-container:has(.landing-hero) .preview-main{padding:17px!important}.block-container:has(.landing-hero) .preview-metrics{grid-template-columns:repeat(3,minmax(0,1fr))!important}.block-container:has(.landing-hero) .preview-metric{padding:10px 8px!important}.block-container:has(.landing-hero) .preview-metric b{font-size:17px!important}}


/* Cogniscan typography system: Poppins only, regular weight throughout. */
html,body,[class*="css"],.stApp,.stApp *,
[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"] *,
[data-testid="stMain"],[data-testid="stMain"] *,
button,input,textarea,select,label,
h1,h2,h3,h4,h5,h6,strong,b,th {
  font-family:'Poppins',sans-serif!important;
  font-weight:400!important;
}


/* Landing alignment and compact primary action refinements */
.block-container:has(.landing-hero) .brandbar{
  justify-content:center!important;
  text-align:center!important;
}
.block-container:has(.landing-hero) .brand{
  justify-content:center!important;
  text-align:center!important;
}
.block-container:has(.landing-hero) .landing-hero{
  width:100%!important;
  align-items:center!important;
  justify-content:center!important;
  text-align:center!important;
  margin-left:auto!important;
  margin-right:auto!important;
}
.block-container:has(.landing-hero) .landing-title,
.block-container:has(.landing-hero) .landing-subtitle{
  width:100%!important;
  text-align:center!important;
  margin-left:auto!important;
  margin-right:auto!important;
}
.block-container:has(.landing-hero) .landing-actions{
  width:auto!important;
  max-width:none!important;
  margin:22px auto 0!important;
  justify-content:center!important;
  align-self:center!important;
}
.block-container:has(.landing-hero) .landing-actions .stButton{
  width:auto!important;
  flex:0 0 auto!important;
}
.block-container:has(.landing-hero) .landing-actions .stButton>button{
  width:auto!important;
  min-width:205px!important;
  min-height:54px!important;
  padding:0 27px!important;
  background:#000!important;
  color:#fff!important;
  border:1px solid #000!important;
  box-shadow:none!important;
  font-size:14px!important;
}
.block-container:has(.landing-hero) .landing-actions .stButton>button:hover{
  background:#222!important;
  border-color:#222!important;
}
.block-container:has(.landing-hero) .dashboard-preview{
  margin-top:24px!important;
  margin-left:auto!important;
  margin-right:auto!important;
}
@media(max-width:850px){
  .block-container:has(.landing-hero) .dashboard-preview{margin-top:20px!important}
}
@media(max-width:560px){
  .block-container:has(.landing-hero) .landing-actions{width:100%!important;margin-top:20px!important}
  .block-container:has(.landing-hero) .landing-actions .stButton{width:auto!important}
  .block-container:has(.landing-hero) .landing-actions .stButton>button{
    width:auto!important;min-width:190px!important;min-height:52px!important;font-size:14px!important
  }
  .block-container:has(.landing-hero) .dashboard-preview{margin-top:18px!important}
}


/* Match Cogniscan's compact black pill CTA. */
.block-container:has(.landing-hero) [data-testid="stButton"]:has(button[kind="primary"]){
  width:fit-content!important;
  max-width:max-content!important;
  margin:22px auto 0!important;
}
.block-container:has(.landing-hero) [data-testid="stButton"]:has(button[kind="primary"]) button{
  width:auto!important;
  min-width:0!important;
  min-height:0!important;
  height:42px!important;
  padding:8px 20px!important;
  border:1px solid #111!important;
  border-radius:9999px!important;
  background:#111!important;
  color:#fff!important;
  box-shadow:none!important;
  font-family:'Poppins',sans-serif!important;
  font-size:13px!important;
  font-weight:400!important;
  letter-spacing:0!important;
  line-height:1.5!important;
  transition:background .2s ease,transform .2s ease!important;
}
.block-container:has(.landing-hero) [data-testid="stButton"]:has(button[kind="primary"]) button:hover{
  background:#000!important;
  border-color:#000!important;
  transform:translateY(-1px)!important;
  box-shadow:none!important;
}
@media(max-width:560px){
  .block-container:has(.landing-hero) [data-testid="stButton"]:has(button[kind="primary"]){
    width:fit-content!important;
    margin:20px auto 0!important;
  }
  .block-container:has(.landing-hero) [data-testid="stButton"]:has(button[kind="primary"]) button{
    width:auto!important;
    min-width:0!important;
    height:40px!important;
    padding:7px 18px!important;
    font-size:12px!important;
  }
}


/* Centered Cogniscan-style CTA. Target the keyed Streamlit element,
   rather than relying on Markdown wrappers around separate Streamlit blocks. */
.block-container:has(.landing-hero) .st-key-start{
  display:flex!important;
  justify-content:center!important;
  align-items:center!important;
  width:100%!important;
  margin:22px auto 26px!important;
}
.block-container:has(.landing-hero) .st-key-start [data-testid="stButton"]{
  display:flex!important;
  justify-content:center!important;
  width:100%!important;
  margin:0!important;
}
.block-container:has(.landing-hero) .st-key-start button{
  display:inline-flex!important;
  align-items:center!important;
  justify-content:center!important;
  width:auto!important;
  min-width:0!important;
  height:48px!important;
  padding:9px 23px!important;
  margin:0 auto!important;
  background:#000!important;
  border:1px solid #000!important;
  border-radius:9999px!important;
  color:#fff!important;
  font-family:'Poppins',sans-serif!important;
  font-size:16px!important;
  font-weight:400!important;
  line-height:1.4!important;
  box-shadow:none!important;
  white-space:nowrap!important;
}
.block-container:has(.landing-hero) .st-key-start button p{
  display:inline!important;
  margin:0!important;
  color:#fff!important;
  font:400 16px/1.4 'Poppins',sans-serif!important;
}
.block-container:has(.landing-hero) .st-key-start button::after{
  content:"↗"!important;
  display:inline-block!important;
  visibility:visible!important;
  opacity:1!important;
  margin-left:11px!important;
  color:#fff!important;
  font-family:Arial,sans-serif!important;
  font-size:22px!important;
  font-weight:400!important;
  line-height:1!important;
}
.block-container:has(.landing-hero) .st-key-start button:hover{
  background:#222!important;
  border-color:#222!important;
  transform:translateY(-1px)!important;
  box-shadow:none!important;
}
@media(max-width:560px){
  .block-container:has(.landing-hero) .st-key-start{margin:20px auto 22px!important}
  .block-container:has(.landing-hero) .st-key-start button{height:46px!important;padding:8px 21px!important}
  .block-container:has(.landing-hero) .st-key-start button p{font-size:15px!important}
}

/* Refine hero vertical position and keep the headline to two lines. */
.block-container:has(.landing-hero) .landing-hero{
  margin-top:clamp(28px,6.5vh,76px)!important;
}
.block-container:has(.landing-hero) .landing-title{
  width:100%!important;
  max-width:1400px!important;
  margin-bottom:12px!important;
}
.block-container:has(.landing-hero) .landing-subtitle{
  max-width:780px!important;
  margin-top:0!important;
}
@media(max-width:850px){
  .block-container:has(.landing-hero) .landing-hero{
    margin-top:clamp(26px,5vh,58px)!important;
  }
  .block-container:has(.landing-hero) .landing-title{
    max-width:100%!important;
  }
}


/* Step 01: full-viewport responsive workflow selection.
   Keep the brandbar aligned with the landing page on every workflow page. */
.block-container:has(.st-key-choose_single){
  display:flex!important;
  flex-direction:column!important;
  width:100%!important;
  max-width:none!important;
  height:100vh!important;
  min-height:100vh!important;
  max-height:100vh!important;
  overflow:hidden!important;
  box-sizing:border-box!important;
  padding:12px clamp(18px,4.5vw,72px) 22px!important;
}
.block-container:has(.st-key-choose_single) .brandbar{
  flex:0 0 auto!important;
  width:min(100%,1420px)!important;
  min-height:38px!important;
  margin:0 auto!important;
  justify-content:center!important;
}
.block-container:has(.st-key-choose_single) .brand{
  justify-content:center!important;
  text-align:center!important;
}
/* Back navigation and step content sit below the fixed brand position. */
.block-container:has(.st-key-choose_single) > div:has(button){
  flex:0 0 auto;
}
.block-container:has(.st-key-choose_single) .hero{
  width:min(100%,900px)!important;
  margin:clamp(10px,2.5vh,28px) auto 26px!important;
  padding:0 12px!important;
  text-align:center!important;
  background:transparent!important;
  border:0!important;
  box-shadow:none!important;
}
.block-container:has(.st-key-choose_single) .hero .eyebrow{
  color:#000!important;
  letter-spacing:.14em!important;
}
.block-container:has(.st-key-choose_single) .hero h1{
  color:#000!important;
  font-size:clamp(32px,4.2vw,54px)!important;
  line-height:1.12!important;
  margin:0 0 12px!important;
}
.block-container:has(.st-key-choose_single) .hero p{
  max-width:620px!important;
  margin:0 auto!important;
  color:#555!important;
  font-size:clamp(13px,1.2vw,15px)!important;
}
.block-container:has(.st-key-choose_single) [data-testid="stHorizontalBlock"]{
  width:min(100%,1040px)!important;
  margin:0 auto!important;
  gap:clamp(14px,2.2vw,26px)!important;
  align-items:stretch!important;
}
.block-container:has(.st-key-choose_single) [data-testid="column"]{
  min-width:0!important;
}
.block-container:has(.st-key-choose_single) .glass{
  display:flex!important;
  flex-direction:column!important;
  justify-content:center!important;
  min-height:clamp(190px,25vh,245px)!important;
  height:100%!important;
  padding:clamp(20px,2.5vw,32px)!important;
  margin:0 0 14px!important;
  color:#000!important;
  background:rgba(255,255,255,.82)!important;
  border:1px solid #d8d8d8!important;
  border-radius:20px!important;
  box-shadow:0 12px 38px rgba(0,0,0,.055)!important;
  backdrop-filter:blur(18px)!important;
}
.block-container:has(.st-key-choose_single) .feature-icon{
  width:42px!important;
  height:42px!important;
  margin-bottom:18px!important;
  color:#fff!important;
  background:#000!important;
  border-radius:13px!important;
}
.block-container:has(.st-key-choose_single) .feature-icon svg{
  stroke:#fff!important;
}
.block-container:has(.st-key-choose_single) .glass h3,
.block-container:has(.st-key-choose_single) .glass .muted,
.block-container:has(.st-key-choose_single) .glass .kicker{
  color:#000!important;
}
.block-container:has(.st-key-choose_single) .glass h3{
  font-size:21px!important;
  margin:0 0 8px!important;
}
.block-container:has(.st-key-choose_single) .glass .muted{
  font-size:13px!important;
  line-height:1.65!important;
}
.block-container:has(.st-key-choose_single) .glass .kicker{
  margin-top:17px!important;
  font-size:9px!important;
  letter-spacing:.13em!important;
}
.block-container:has(.st-key-choose_single) .st-key-choose_single,
.block-container:has(.st-key-choose_single) .st-key-choose_many{
  display:flex!important;
  justify-content:center!important;
  width:100%!important;
  margin:0!important;
}
.block-container:has(.st-key-choose_single) .st-key-choose_single button,
.block-container:has(.st-key-choose_single) .st-key-choose_many button{
  width:100%!important;
  min-height:48px!important;
  color:#fff!important;
  background:#000!important;
  border:1px solid #000!important;
  border-radius:999px!important;
  box-shadow:none!important;
  font-family:'Poppins',sans-serif!important;
  font-size:13px!important;
  font-weight:400!important;
}
.block-container:has(.st-key-choose_single) .st-key-choose_single button:hover,
.block-container:has(.st-key-choose_single) .st-key-choose_many button:hover{
  color:#000!important;
  background:#fff!important;
  border-color:#000!important;
}
@media(max-width:760px){
  .block-container:has(.st-key-choose_single){
    padding:12px 18px 18px!important;
  }
  .block-container:has(.st-key-choose_single) .hero{
    margin:clamp(10px,2.5vh,24px) auto 18px!important;
  }
  .block-container:has(.st-key-choose_single) [data-testid="stHorizontalBlock"]{
    gap:12px!important;
  }
  .block-container:has(.st-key-choose_single) .glass{
    min-height:clamp(165px,22vh,205px)!important;
    padding:18px!important;
  }
}
@media(max-width:560px){
  .block-container:has(.st-key-choose_single){
    height:100svh!important;
    min-height:100svh!important;
    max-height:100svh!important;
    padding:10px 14px 14px!important;
  }
  .block-container:has(.st-key-choose_single) .hero{
    margin:12px auto 16px!important;
    padding:0 4px!important;
  }
  .block-container:has(.st-key-choose_single) .hero h1{
    font-size:clamp(29px,8vw,38px)!important;
  }
  .block-container:has(.st-key-choose_single) .hero p{
    font-size:12px!important;
    line-height:1.55!important;
  }
  .block-container:has(.st-key-choose_single) [data-testid="stHorizontalBlock"]{
    flex-direction:column!important;
    gap:12px!important;
  }
  .block-container:has(.st-key-choose_single) [data-testid="column"]{
    width:100%!important;
    flex:1 1 auto!important;
  }
  .block-container:has(.st-key-choose_single) .glass{
    min-height:0!important;
    padding:15px 17px!important;
    margin-bottom:9px!important;
  }
  .block-container:has(.st-key-choose_single) .feature-icon{
    width:34px!important;
    height:34px!important;
    margin-bottom:10px!important;
  }
  .block-container:has(.st-key-choose_single) .glass h3{
    font-size:17px!important;
    margin-bottom:5px!important;
  }
  .block-container:has(.st-key-choose_single) .glass .muted{
    font-size:11px!important;
    line-height:1.45!important;
  }
  .block-container:has(.st-key-choose_single) .glass .kicker{
    margin-top:9px!important;
  }
}

/* Monochrome dashboard preview with the landing-page mark. */
.block-container:has(.landing-hero) .dashboard-preview{
  color:#000!important;
  background:rgba(255,255,255,.78)!important;
  border-color:#111!important;
}
.block-container:has(.landing-hero) .preview-sidebar,
.block-container:has(.landing-hero) .preview-main,
.block-container:has(.landing-hero) .preview-metric,
.block-container:has(.landing-hero) .preview-panel{
  color:#000!important;
  background:#fff!important;
  border-color:#111!important;
  box-shadow:none!important;
}
.block-container:has(.landing-hero) .preview-brand{
  color:#000!important;
}
.block-container:has(.landing-hero) .preview-mark{
  display:grid!important;
  place-items:center!important;
  color:#fff!important;
  background:#000!important;
  border-radius:7px!important;
}
.block-container:has(.landing-hero) .preview-mark svg{
  display:block!important;
  width:16px!important;
  height:16px!important;
}
.block-container:has(.landing-hero) .preview-nav,
.block-container:has(.landing-hero) .preview-nav.active,
.block-container:has(.landing-hero) .preview-head strong,
.block-container:has(.landing-hero) .preview-head span,
.block-container:has(.landing-hero) .preview-metric small,
.block-container:has(.landing-hero) .preview-metric b,
.block-container:has(.landing-hero) .preview-metric em,
.block-container:has(.landing-hero) .preview-panel-title,
.block-container:has(.landing-hero) .sentiment-row,
.block-container:has(.landing-hero) .sentiment-row b{
  color:#000!important;
}
.block-container:has(.landing-hero) .preview-nav.active{
  background:#eee!important;
}
.block-container:has(.landing-hero) .preview-head{
  align-items:flex-start!important;
}
.block-container:has(.landing-hero) .preview-head > div{
  order:2!important;
  margin-left:auto!important;
  text-align:right!important;
}
.block-container:has(.landing-hero) .preview-head > span{
  order:1!important;
  margin-right:auto!important;
  color:#000!important;
}
.block-container:has(.landing-hero) .preview-head strong{
  display:inline-block!important;
  font-size:21px!important;
  line-height:1.3!important;
}
.block-container:has(.landing-hero) .preview-metric em{
  color:#000!important;
}
.block-container:has(.landing-hero) .mini-bars{
  border-bottom-color:#000!important;
  background:repeating-linear-gradient(to bottom,transparent 0,transparent 26px,#e5e5e5 27px)!important;
}
.block-container:has(.landing-hero) .mini-bars i{
  background:#000!important;
  opacity:1!important;
}
.block-container:has(.landing-hero) .sentiment-dot{
  background:#000!important;
}
.block-container:has(.landing-hero) .preview-mark,
.block-container:has(.landing-hero) .preview-metric,
.block-container:has(.landing-hero) .preview-panel{
  border-color:#d4d4d4!important;
}



/* Use the browser's currently visible viewport, not the legacy layout viewport.
   Apply the same viewport sizing to the landing and Step 01 screens. */
@supports (height:100dvh){
  [data-testid="stMain"]:has(.landing-hero),
  [data-testid="stMain"]:has(.st-key-choose_single){
    height:100dvh!important;
    min-height:100dvh!important;
    max-height:100dvh!important;
    overflow:hidden!important;
  }
  [data-testid="stMain"]:has(.landing-hero) [data-testid="stMainBlockContainer"],
  [data-testid="stMain"]:has(.st-key-choose_single) [data-testid="stMainBlockContainer"]{
    height:100%!important;
    min-height:0!important;
    max-height:100%!important;
    overflow:hidden!important;
  }
  .block-container:has(.landing-hero),
  .block-container:has(.st-key-choose_single){
    height:100dvh!important;
    min-height:100dvh!important;
    max-height:100dvh!important;
    overflow:hidden!important;
    box-sizing:border-box!important;
  }
}
@supports not (height:100dvh){
  [data-testid="stMain"]:has(.landing-hero),
  [data-testid="stMain"]:has(.st-key-choose_single),
  .block-container:has(.landing-hero),
  .block-container:has(.st-key-choose_single){
    height:100svh!important;
    min-height:100svh!important;
    max-height:100svh!important;
    overflow:hidden!important;
    box-sizing:border-box!important;
  }
}

/* Step 02 — focused input workspace */
.block-container:has(.st-key-analyze_single),.block-container:has(.st-key-analyze_batch){max-width:1040px!important;padding:18px clamp(18px,4vw,54px) 28px!important}
.block-container:has(.st-key-analyze_single) .brandbar,.block-container:has(.st-key-analyze_batch) .brandbar{max-width:100%;margin:0 auto 18px!important}
.step2-progress{max-width:860px;margin:0 auto 15px;padding:0 2px}
.step2-progress>div:first-child{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;color:#303030;font-size:10px;letter-spacing:.12em}
.step2-progress>div:first-child span:first-child{color:#777;letter-spacing:.16em}.step2-progress i{font-style:normal;color:#999}
.step2-track{height:3px;border-radius:8px;background:#e9e9e9;overflow:hidden}.step2-track span{display:block;width:100%;height:100%;background:#171717;border-radius:8px}
.block-container:has(.st-key-analyze_single) .hero,.block-container:has(.st-key-analyze_batch) .hero{max-width:860px;margin:0 auto 17px;padding:25px 30px 23px;border:1px solid #e9e9e9;border-radius:20px;background:rgba(255,255,255,.82);box-shadow:0 12px 34px rgba(0,0,0,.035);backdrop-filter:blur(18px)}
.block-container:has(.st-key-analyze_single) .hero:after,.block-container:has(.st-key-analyze_batch) .hero:after{display:none}
.block-container:has(.st-key-analyze_single) .hero .eyebrow,.block-container:has(.st-key-analyze_batch) .hero .eyebrow{color:#777;font-size:10px;letter-spacing:.16em;font-weight:400;margin-bottom:9px}
.block-container:has(.st-key-analyze_single) .hero h1,.block-container:has(.st-key-analyze_batch) .hero h1{color:#171717;font-size:clamp(30px,4vw,43px);font-weight:400;letter-spacing:-.055em;line-height:1.12;margin-bottom:8px}
.block-container:has(.st-key-analyze_single) .hero p,.block-container:has(.st-key-analyze_batch) .hero p{color:#777;font-size:12px;line-height:1.7;max-width:650px}
.block-container:has(.st-key-analyze_single) .glass,.block-container:has(.st-key-analyze_batch) .glass{max-width:860px;margin:0 auto 10px;padding:19px 22px;background:rgba(255,255,255,.88);border:1px solid #e9e9e9;border-radius:17px;box-shadow:0 8px 26px rgba(0,0,0,.025);backdrop-filter:blur(18px)}
.block-container:has(.st-key-analyze_single) .glass h3,.block-container:has(.st-key-analyze_batch) .glass h3{color:#222;font-size:15px;font-weight:400;letter-spacing:-.02em;margin-bottom:4px}
.block-container:has(.st-key-analyze_single) .glass .kicker,.block-container:has(.st-key-analyze_batch) .glass .kicker{color:#888;font-size:9px;font-weight:400;letter-spacing:.16em;margin-bottom:7px}
.block-container:has(.st-key-analyze_single) .glass .muted,.block-container:has(.st-key-analyze_batch) .glass .muted{color:#858585;font-size:11px}
.block-container:has(.st-key-analyze_single) [data-testid="stTextArea"],.block-container:has(.st-key-analyze_batch) [data-testid="stTextArea"]{max-width:860px;margin:0 auto!important;padding:0 2px}
.block-container:has(.st-key-analyze_single) .stTextArea textarea,.block-container:has(.st-key-analyze_batch) .stTextArea textarea{min-height:190px!important;padding:17px 18px!important;border:1px solid #dedede!important;border-radius:15px!important;background:rgba(255,255,255,.94)!important;color:#222!important;font:400 13px/1.8 Poppins,sans-serif!important;box-shadow:0 7px 22px rgba(0,0,0,.025)!important;resize:vertical}
.block-container:has(.st-key-analyze_single) .stTextArea textarea:focus,.block-container:has(.st-key-analyze_batch) .stTextArea textarea:focus{border-color:#777!important;box-shadow:0 0 0 3px rgba(0,0,0,.045)!important}
.block-container:has(.st-key-analyze_single) [data-testid="stCaptionContainer"]{max-width:860px;margin:3px auto 13px;color:#888;font-size:10px}
.block-container:has(.st-key-analyze_single) .st-key-analyze_single,.block-container:has(.st-key-analyze_batch) .st-key-analyze_batch{max-width:860px;margin:0 auto}
.block-container:has(.st-key-analyze_single) .st-key-analyze_single button,.block-container:has(.st-key-analyze_batch) .st-key-analyze_batch button{min-height:48px!important;border-radius:13px!important;background:#171717!important;color:#fff!important;border:1px solid #171717!important;box-shadow:0 8px 20px rgba(0,0,0,.10)!important;font-weight:400!important}
.block-container:has(.st-key-analyze_single) .st-key-analyze_single button:hover,.block-container:has(.st-key-analyze_batch) .st-key-analyze_batch button:hover{background:#333!important;transform:translateY(-1px)}
.block-container:has(.st-key-analyze_batch) [data-testid="stFileUploader"]{max-width:860px;margin:12px auto 14px;padding:15px 17px;border:1px solid #e5e5e5;border-radius:15px;background:rgba(255,255,255,.78)}
.block-container:has(.st-key-analyze_batch) [data-testid="stFileUploader"] section{border:1px dashed #c9c9c9!important;background:#fafafa!important;border-radius:11px!important}
.block-container:has(.st-key-analyze_batch) [data-testid="stSelectbox"]{max-width:860px;margin:0 auto 12px}
@media(max-width:700px){.block-container:has(.st-key-analyze_single),.block-container:has(.st-key-analyze_batch){padding:12px 14px 24px!important}.block-container:has(.st-key-analyze_single) .hero,.block-container:has(.st-key-analyze_batch) .hero{padding:21px 19px;border-radius:17px}.block-container:has(.st-key-analyze_single) .glass,.block-container:has(.st-key-analyze_batch) .glass{padding:16px 17px}.block-container:has(.st-key-analyze_single) .stTextArea textarea,.block-container:has(.st-key-analyze_batch) .stTextArea textarea{min-height:170px!important}}

/* Compact liquid-style sentiment donut and balanced result cards */
.sentiment-donut-wrap{width:100%;display:flex;align-items:center;justify-content:center;padding:2px 0 8px}
.sentiment-donut{display:block;width:min(100%,228px);height:auto;overflow:visible;filter:drop-shadow(0 8px 15px rgba(40,45,65,.08))}
.donut-center-value{font:400 25px Poppins,sans-serif;fill:#202020;letter-spacing:-.045em}
.donut-center-label{font:400 9px Poppins,sans-serif;fill:#888;letter-spacing:.15em}
.block-container:has(.st-key-back_input) .hero{margin-bottom:16px!important}
.block-container:has(.st-key-back_input) [data-testid="stHorizontalBlock"]{align-items:stretch}
.block-container:has(.st-key-back_input) [data-testid="stColumn"]{min-width:0}
.block-container:has(.st-key-back_input) .glass{height:auto;min-height:0}
.block-container:has(.st-key-back_input) .glass:has(.sentiment-donut-wrap){display:flex;flex-direction:column}
.block-container:has(.st-key-back_input) .glass:has(.sentiment-donut-wrap) h3{margin-bottom:4px}
.block-container:has(.st-key-back_input) .glass:has(.sentiment-donut-wrap) .sentiment-donut-wrap{flex:1;min-height:215px}
.block-container:has(.st-key-back_input) .footer{display:none!important}
@media(max-width:760px){.sentiment-donut{width:min(100%,205px)}.block-container:has(.st-key-back_input) .glass:has(.sentiment-donut-wrap) .sentiment-donut-wrap{min-height:195px}}


/* Keep the polarity card aligned with the overall-sentiment card */
.block-container:has(.st-key-back_input) [data-testid="stHorizontalBlock"]{
  align-items:flex-start!important;
}
.block-container:has(.st-key-back_input) [data-testid="column"]:first-child .glass:first-of-type,
.block-container:has(.st-key-back_input) .glass:has(.sentiment-donut-wrap){
  height:226px!important;
  min-height:226px!important;
  max-height:226px!important;
  box-sizing:border-box!important;
  overflow:hidden!important;
}
.block-container:has(.st-key-back_input) .glass:has(.sentiment-donut-wrap){
  display:flex!important;
  flex-direction:column!important;
  justify-content:flex-start!important;
}
.block-container:has(.st-key-back_input) .glass:has(.sentiment-donut-wrap) .sentiment-donut-wrap{
  flex:1 1 auto!important;
  min-height:0!important;
  padding:0!important;
  margin-top:-2px!important;
}
.block-container:has(.st-key-back_input) .sentiment-donut{
  width:min(100%,158px)!important;
  max-height:158px!important;
}
.block-container:has(.st-key-back_input) .glass:has(.sentiment-donut-wrap) h3{
  margin:0 0 2px!important;
  font-size:15px!important;
}
.block-container:has(.st-key-back_input) .glass:has(.sentiment-donut-wrap) p{
  margin-bottom:2px!important;
  font-size:10px!important;
  line-height:1.45!important;
}
@media(max-width:760px){
  .block-container:has(.st-key-back_input) [data-testid="column"]:first-child .glass:first-of-type,
  .block-container:has(.st-key-back_input) .glass:has(.sentiment-donut-wrap){
    height:auto!important;min-height:0!important;max-height:none!important;
  }
  .block-container:has(.st-key-back_input) .glass:has(.sentiment-donut-wrap) .sentiment-donut-wrap{
    min-height:0!important;padding:8px 0!important;
  }
  .block-container:has(.st-key-back_input) .sentiment-donut{width:min(100%,175px)!important}
}


/* Standalone large donut: remove the enclosing polarity card */
.block-container:has(.st-key-back_input) .sentiment-donut-wrap{
  min-height:0!important;
  padding:8px 0!important;
}
.block-container:has(.st-key-back_input) .sentiment-donut{
  width:min(100%,310px)!important;
  max-height:none!important;
  filter:drop-shadow(0 14px 24px rgba(40,45,65,.10));
}
@media(max-width:760px){
  .block-container:has(.st-key-back_input) .sentiment-donut{width:min(100%,260px)!important}
}

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
    """Build a compact SVG donut with rounded segment caps."""
    labels = ["Positive", "Neutral", "Negative"]
    gradients = [("#45C69A", "#168D70"), ("#F4C875", "#D99A3D"), ("#F18A98", "#D95D70")]
    total = sum(max(0.0, float(values.get(label, 0))) for label in labels) or 1.0
    radius = 78
    circumference = 2 * 3.141592653589793 * radius
    gap = 3.5
    offset = 0.0
    segments = []
    for index, label in enumerate(labels):
        amount = max(0.0, float(values.get(label, 0)))
        if amount <= 0:
            continue
        arc = circumference * amount / total
        visible_arc = max(0.0, arc - gap)
        segments.append(
            f'<circle cx="110" cy="110" r="{radius}" fill="none" '
            f'stroke="url(#sentimentGradient{index})" stroke-width="23" '
            f'stroke-linecap="round" stroke-dasharray="{visible_arc:.3f} {circumference-visible_arc:.3f}" '
            f'stroke-dashoffset="{-offset:.3f}" />'
        )
        offset += arc

    svg = f"""
    <div class="sentiment-donut-wrap">
      <svg class="sentiment-donut" viewBox="0 0 220 220" role="img" aria-label="Sentiment distribution">
        <defs>
          <linearGradient id="sentimentGradient0" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="{gradients[0][0]}"/><stop offset="100%" stop-color="{gradients[0][1]}"/></linearGradient>
          <linearGradient id="sentimentGradient1" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="{gradients[1][0]}"/><stop offset="100%" stop-color="{gradients[1][1]}"/></linearGradient>
          <linearGradient id="sentimentGradient2" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="{gradients[2][0]}"/><stop offset="100%" stop-color="{gradients[2][1]}"/></linearGradient>
        </defs>
        <circle cx="110" cy="110" r="{radius}" fill="none" stroke="#F0F0F0" stroke-width="23"/>
        <g transform="rotate(-90 110 110)">{"".join(segments)}</g>
        <text x="110" y="105" text-anchor="middle" class="donut-center-value">{center_value}</text>
        <text x="110" y="126" text-anchor="middle" class="donut-center-label">{center_label}</text>
      </svg>
    </div>
    """
    return svg

def brandbar():
    st.markdown("""
    <div class="brandbar">
      <div class="brand"><div class="brand-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg></div>TERNA SENTIMENT ANALYZER</div>
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
    <section class="landing-hero">
      <h1 class="landing-title">Understand Student Feedback.<br>See the Sentiment Across Your College.</h1>
      <p class="landing-subtitle">Analyze sentiments across your college feedback dataset and understand what students are saying, what is working, and where improvements are needed.</p>
    </section>
    """, unsafe_allow_html=True)
    st.markdown('<div class="landing-actions">', unsafe_allow_html=True)
    if st.button("Start analyzing", type="primary", use_container_width=False, key="start"):
        go("choose")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <section class="dashboard-preview">
      <div class="preview-shell">
        <aside class="preview-sidebar">
          <div class="preview-brand"><span class="preview-mark"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg></span> TERNA SENTIMENT ANALYZER</div>
          <div class="preview-nav active">▦ &nbsp; Overview</div>
          <div class="preview-nav">▤ &nbsp; Single analysis</div>
          <div class="preview-nav">▧ &nbsp; Batch reviews</div>
          <div class="preview-nav">⌁ &nbsp; Sentiment trends</div>
          <div class="preview-nav">⚙ &nbsp; Settings</div>
        </aside>
        <div class="preview-main">
          <div class="preview-head"><div><strong>College feedback overview</strong><br><span>A snapshot of student sentiment across your institution</span></div><span>SAMPLE DASHBOARD &nbsp; ◉</span></div>
          <div class="preview-metrics">
            <div class="preview-metric"><small>Responses analyzed</small><b>2,480</b><em>↑ Student feedback</em></div>
            <div class="preview-metric"><small>Positive sentiment</small><b>68.4%</b><em>↑ Positive experience</em></div>
            <div class="preview-metric"><small>Negative sentiment</small><b>12.6%</b><em>↓ Needs attention</em></div>
          </div>
          <div class="preview-bottom">
            <div class="preview-panel"><div class="preview-panel-title">Feedback sentiment</div><div class="mini-bars"><i style="height:30%"></i><i style="height:44%"></i><i style="height:39%"></i><i style="height:61%"></i><i style="height:52%"></i><i style="height:73%"></i><i style="height:66%"></i><i style="height:88%"></i><i style="height:78%"></i><i style="height:96%"></i><i style="height:84%"></i><i style="height:100%"></i></div></div>
            <div class="preview-panel"><div class="preview-panel-title">Sentiment distribution</div><div class="sentiment-legend">
              <div class="sentiment-row"><span class="sentiment-dot" style="background:#27a982"></span>Positive <b>68.4%</b></div>
              <div class="sentiment-row"><span class="sentiment-dot" style="background:#5683ef"></span>Neutral <b>19.0%</b></div>
              <div class="sentiment-row"><span class="sentiment-dot" style="background:#e16d79"></span>Negative <b>12.6%</b></div>
            </div></div>
          </div>
        </div>
      </div>
    </section>
    <div class="footer">TERNA SENTIMENT ANALYZER · POWERED BY NLTK VADER</div>
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
        st.markdown('<div class="step2-progress"><div><span>WORKFLOW</span><span>02 <i>/ 02</i></span></div><div class="step2-track"><span></span></div></div><div class="hero"><div class="eyebrow">STEP 02 · SINGLE ANALYSIS</div><h1>Paste your text.</h1><p>Enter a review, message or comment. The analyzer will identify its overall sentiment and polarity distribution.</p></div>', unsafe_allow_html=True)
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
        st.markdown('<div class="step2-progress"><div><span>WORKFLOW</span><span>02 <i>/ 02</i></span></div><div class="step2-track"><span></span></div></div><div class="hero"><div class="eyebrow">STEP 02 · BATCH ANALYSIS</div><h1>Bring your feedback.</h1><p>Paste one review per line or import a CSV file. You can choose the column containing your feedback before analysis.</p></div>', unsafe_allow_html=True)
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
            st.markdown(f'<div style="font:400 29px Poppins,sans-serif;color:{color};margin:7px 0">{result["label"]}</div><div class="muted">Compound score</div><div class="score">{scores["compound"]:+.3f}</div><div class="muted">Scale: −1.0 (negative) to +1.0 (positive)</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="glass"><div class="kicker">ANALYZED TEXT</div>', unsafe_allow_html=True)
            st.write(result["text"])
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown(donut_chart({"Positive":scores["pos"],"Neutral":scores["neu"],"Negative":scores["neg"]},"POLARITY","100%"), unsafe_allow_html=True)
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
            st.markdown(donut_chart(summary,"TOTAL",str(total)), unsafe_allow_html=True)
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

