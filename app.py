
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================
# RETAILIQ AI - BUSINESS INTELLIGENCE APP
# ==========================================

st.set_page_config(
    page_title="RetailIQ AI",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# RETAILIQ AI - CUSTOM UI THEME
# ==========================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #F5F7FA;
    color: #14213D;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #14213D;
}

/* Batch 12.1: keep sidebar panels and controls readable on the navy background. */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: #1E3153 !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary *,
[data-testid="stSidebar"] [data-testid="stExpander"] p,
[data-testid="stSidebar"] [data-testid="stExpander"] span,
[data-testid="stSidebar"] [data-testid="stExpander"] label,
[data-testid="stSidebar"] [data-testid="stExpander"] div {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMetric"] {
    background-color: #263B60 !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMetricLabel"],
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Fix success/status cards inside the readiness expander.
   Streamlit success alerts have a pale-green default background, so use
   a dark navy panel with high-contrast white text in the sidebar. */
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"],
[data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="notification"] {
    background: #263B60 !important;
    background-color: #263B60 !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    border-radius: 9px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"] *,
[data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="notification"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button,
[data-testid="stSidebar"] .stDownloadButton button {
    background-color: #00A6A6 !important;
    color: #FFFFFF !important;
    border: 1px solid #00A6A6 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #14213D !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section * {
    color: #14213D !important;
}


/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #14213D;
}

/* Keep sidebar headings and labels white */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown {
    color: #FFFFFF;
}

/* Currency selector - dark readable text */
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #14213D !important;
}

/* File uploader - readable text */
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: #14213D !important;
}

/* File uploader label */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: #FFFFFF;
    border-radius: 12px;
}

/* Sidebar information box */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: #E8F1FF;
    border-radius: 12px;
}

[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #14213D !important;
}

/* Navigation labels */
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #FFFFFF !important;
}


/* Main headings */
h1, h2, h3 {
    color: #14213D !important;
    font-weight: 700 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5EAF1;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(20, 33, 61, 0.06);
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #00A6A6 !important;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background-color: #00A6A6;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #008B8B;
    color: white;
    border: none;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #E5EAF1;
    border-radius: 10px;
}

/* Input fields */
.stTextInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 8px;
}

/* Info and success messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.main {
    background-color: #F7F9FC;
}
.block-container {
    padding-top: 1.5rem;
}
h1, h2, h3 {
    color: #172554;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Batch 11 — visual consistency and responsive dashboard polish */
.block-container {
    max-width: 1500px;
    padding-left: clamp(1rem, 3vw, 2.5rem);
    padding-right: clamp(1rem, 3vw, 2.5rem);
}
h1, h2, h3 {
    letter-spacing: -0.025em;
}
[data-testid="stMetric"] {
    min-height: 112px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.82rem;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    font-size: clamp(1.35rem, 2vw, 2rem);
}
[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid #E5EAF1;
    border-radius: 14px;
    padding: 0.35rem;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background: #FFFFFF;
    border-radius: 12px;
}
.stDownloadButton > button {
    border: 1px solid #00A6A6;
    border-radius: 9px;
    font-weight: 600;
}
[data-testid="stExpander"] {
    border: 1px solid #E5EAF1;
    border-radius: 12px;
    background: #FFFFFF;
}
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.35rem;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 9px;
    padding: 0.35rem 0.5rem;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
    min-height: 88px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.16);
    box-shadow: none;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"],
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}
.workspace-kicker {
    color: #0F8F8F;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: -0.45rem;
}
.workspace-subtitle {
    color: #64748B;
    font-size: 0.92rem;
    margin-top: -0.55rem;
}
@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    [data-testid="stMetric"] {
        min-height: 90px;
        padding: 12px;
    }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
:root {
  --ri-bg: #080811;
  --ri-panel: #11111f;
  --ri-panel-2: #19182a;
  --ri-border: rgba(180, 112, 255, .25);
  --ri-purple: #9b4dff;
  --ri-violet: #c46bff;
  --ri-cyan: #4de8ed;
  --ri-muted: #a9a2bd;
  --ri-white: #f7f4ff;
}
.stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at 72% 8%, rgba(94, 37, 168, .18), transparent 34%),
    radial-gradient(ellipse at 5% 90%, rgba(41, 77, 170, .12), transparent 30%),
    var(--ri-bg) !important;
  color: var(--ri-white) !important;
}
.block-container {
  max-width: 1700px;
  padding-top: 1.25rem;
  padding-bottom: 3rem;
}
h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
  color: var(--ri-white);
}
h1, h2, h3 { letter-spacing: -.035em; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b0a14 0%, #111020 100%) !important;
  border-right: 1px solid rgba(160, 95, 255, .2);
}
[data-testid="stSidebar"] * { color: #f3edff; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
  background: #171427 !important;
  border: 1px dashed rgba(184, 119, 255, .65) !important;
  border-radius: 14px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"],
[data-testid="stExpander"] {
  background: #151323 !important;
  border: 1px solid var(--ri-border) !important;
  border-radius: 14px !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
  background: #201a35 !important;
  border: 1px solid rgba(173, 111, 255, .25) !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] * { color: #f7f1ff !important; }
[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: #1a1730 !important;
  border: 1px solid var(--ri-border) !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"],
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #fff !important; }
[data-testid="stMetric"] {
  background: linear-gradient(145deg, #191629, #11111e) !important;
  border: 1px solid rgba(170, 111, 255, .22) !important;
  border-radius: 17px !important;
  box-shadow: 0 8px 26px rgba(0,0,0,.22) !important;
  padding: 18px !important;
}
[data-testid="stMetricLabel"] { color: #b8afcb !important; }
[data-testid="stMetricValue"] { color: #d5a0ff !important; font-weight: 750 !important; }
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(110deg, #7c2cff, #b14cff) !important;
  color: #fff !important;
  border: 1px solid rgba(220, 180, 255, .25) !important;
  border-radius: 11px !important;
  font-weight: 700 !important;
  box-shadow: 0 5px 22px rgba(142, 61, 255, .22);
  transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 28px rgba(160, 75, 255, .4);
  filter: brightness(1.12);
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--ri-border) !important;
  border-radius: 12px !important;
}
[data-testid="stPlotlyChart"] {
  background: #11111d !important;
  border: 1px solid var(--ri-border) !important;
  border-radius: 16px !important;
  padding: .35rem;
}
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input,
.stDateInput input, .stMultiSelect [data-baseweb="select"] {
  background: #171527 !important;
  color: #f8f4ff !important;
  border-color: rgba(174, 118, 255, .35) !important;
}
[data-testid="stFileUploader"] section {
  background: linear-gradient(120deg, rgba(104,48,170,.16), rgba(29,39,73,.3)) !important;
  border: 1px dashed rgba(190, 128, 255, .72) !important;
  border-radius: 17px !important;
}
[data-testid="stFileUploader"] section * { color: #e9dcff !important; }
[data-testid="stAlert"] {
  background: #1b1830 !important;
  border: 1px solid rgba(172, 111, 255, .22) !important;
  border-radius: 13px !important;
}
[data-testid="stAlert"] * { color: #eee7ff !important; }
[data-testid="stRadio"] label { color: #eee7ff !important; }
[data-testid="stExpander"] summary { color: #f7f2ff !important; }
hr { border-color: rgba(177, 119, 255, .18) !important; }
.ri-brand {
  font-size: 1.18rem; font-weight: 850; letter-spacing: -.04em;
  color: #fff; padding: .25rem 0 .8rem;
}
.ri-brand span { color: #c16cff; }
.ri-kicker {
  display: inline-flex; align-items: center; gap: .45rem;
  border: 1px solid rgba(177,111,255,.34);
  background: rgba(123,69,190,.14); border-radius: 999px;
  padding: .42rem .8rem; color: #d9c5ff; font-size: .78rem;
  letter-spacing: .04em; margin-bottom: 1rem;
}
.ri-hero-title {
  font-size: clamp(2.25rem, 4vw, 4.1rem); line-height: 1.04;
  letter-spacing: -.055em; font-weight: 850; color: #fbf8ff;
  margin: .1rem 0 .9rem;
}
.ri-hero-title span {
  background: linear-gradient(95deg, #a66aff, #e18bff 55%, #6de8f2);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.ri-subtitle { color: #b8b0ca; font-size: 1.02rem; line-height: 1.65; max-width: 720px; }
.ri-orb-wrap { display:flex; justify-content:center; align-items:center; min-height:280px; }
.ri-orb {
  width: min(270px, 70vw); aspect-ratio:1; border-radius:50%;
  background:
    radial-gradient(circle at 34% 28%, rgba(255,255,255,.95) 0 1px, transparent 2px),
    radial-gradient(circle at 65% 60%, rgba(211,111,255,.85) 0 1px, transparent 2px),
    repeating-radial-gradient(circle at 50% 50%, transparent 0 8px, rgba(155,74,255,.23) 9px 10px),
    radial-gradient(circle at 35% 30%, #7b31d8, #28104e 55%, #0a0913 75%);
  box-shadow: 0 0 25px #9d4dff, 0 0 65px rgba(142,55,255,.55), inset 0 0 38px rgba(220,140,255,.5);
  position:relative; animation: ri-float 5s ease-in-out infinite;
}
.ri-orb:before, .ri-orb:after {
  content:""; position:absolute; inset:10%; border-radius:50%;
  border:1px solid rgba(216,161,255,.42); transform:rotate(-25deg) scaleY(.42);
}
.ri-orb:after { inset:-5%; transform:rotate(28deg) scaleY(.34); border-color:rgba(84,218,255,.4); }
@keyframes ri-float { 0%,100% { transform:translateY(0) scale(1); } 50% { transform:translateY(-10px) scale(1.025); } }
.ri-glass {
  background: linear-gradient(145deg, rgba(34,28,54,.88), rgba(17,16,29,.92));
  border: 1px solid rgba(174,111,255,.27); border-radius:18px;
  box-shadow: 0 12px 32px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.04);
}
.ri-module {
  min-height: 158px; padding: 1rem; transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}
.ri-module:hover {
  transform: translateY(-5px); border-color: rgba(197,127,255,.8);
  box-shadow: 0 0 24px rgba(151,72,255,.24);
}
.ri-module-icon {
  width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(140deg,rgba(161,76,255,.35),rgba(64,203,255,.12));
  color:#d7a2ff;font-size:1.35rem;margin-bottom:.8rem;
  box-shadow:0 0 18px rgba(160,81,255,.22);
}
.ri-module-title { font-weight:800;color:#fff;font-size:.98rem;margin-bottom:.35rem; }
.ri-module-copy { color:#b7aec9;font-size:.79rem;line-height:1.45; }
.ri-panel-title { font-size:1.04rem;font-weight:800;color:#fbf8ff;margin-bottom:.25rem; }
.ri-muted { color:#a9a1bb;font-size:.8rem; }
.ri-mini-row { display:flex;justify-content:space-between;gap:1rem;padding:.65rem 0;border-bottom:1px solid rgba(180,130,240,.12);color:#d9d2e8;font-size:.82rem; }
.ri-ready { color:#66f0c0;font-weight:700; }
@media(max-width:900px) {
  .ri-orb-wrap { min-height:190px; }
  .ri-orb { width:190px; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""<style>
/* Refined typography, motion and landing-page information panels */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {
  font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
}
h1, h2, h3, .ri-hero-title, .ri-module-title {
  font-family: 'Space Grotesk', 'DM Sans', sans-serif !important;
}
.ri-hero-title, .ri-kicker, .ri-subtitle, .ri-glass, .ri-module,
[data-testid="stMetric"] {
  animation: ri-enter .65s cubic-bezier(.2,.7,.2,1) both;
}
@keyframes ri-enter {
  from { opacity:0; transform:translateY(12px); }
  to { opacity:1; transform:translateY(0); }
}
.ri-module { position:relative; overflow:hidden; }
.ri-module:before {
  content:""; position:absolute; inset:-1px; opacity:0;
  background:linear-gradient(115deg, transparent 20%, rgba(192,130,255,.12), transparent 75%);
  transform:translateX(-100%); transition:opacity .25s ease;
}
.ri-module:hover:before { opacity:1; animation:ri-shimmer 1.1s ease; }
@keyframes ri-shimmer { to { transform:translateX(100%); } }
.ri-feature-list { display:grid; grid-template-columns:1fr 1fr; gap:.6rem .9rem; }
.ri-feature {
  display:flex; gap:.6rem; align-items:flex-start; color:#e7def7;
  font-size:.84rem; line-height:1.45; padding:.25rem 0;
}
.ri-feature b { color:#fff; font-weight:700; }
.ri-feature-icon {
  display:inline-flex; flex:0 0 25px; width:25px; height:25px;
  align-items:center; justify-content:center; border-radius:8px;
  color:#e5c6ff; background:rgba(153,82,255,.2);
  box-shadow:0 0 12px rgba(153,82,255,.15);
}
.ri-step {
  display:flex; gap:.8rem; align-items:center; padding:.9rem;
  border:1px solid rgba(174,111,255,.24); border-radius:15px;
  background:linear-gradient(145deg,rgba(34,28,54,.82),rgba(17,16,29,.9));
  min-height:88px; transition:transform .2s ease,border-color .2s ease;
}
.ri-step:hover { transform:translateY(-3px); border-color:rgba(198,133,255,.7); }
.ri-step-number {
  display:flex; align-items:center; justify-content:center; flex:0 0 34px;
  width:34px;height:34px;border-radius:50%;font-weight:800;color:#fff;
  background:linear-gradient(135deg,#7b2cff,#c45cff);
  box-shadow:0 0 16px rgba(157,77,255,.4);
}
.ri-step-title { color:#fff;font-weight:800;font-size:.9rem; }
.ri-step-copy { color:#aaa1be;font-size:.76rem;line-height:1.4;margin-top:.15rem; }
.ri-info-panel {
  padding:1.1rem 1.15rem; height:100%;
  background:linear-gradient(145deg,rgba(30,25,48,.9),rgba(15,15,27,.94));
  border:1px solid rgba(174,111,255,.25); border-radius:17px;
}
.ri-info-title { color:#fff;font-size:1rem;font-weight:800;margin-bottom:.65rem; }
.ri-muted { color:#aaa2bc !important; }
@media(max-width:800px) {
  .ri-feature-list { grid-template-columns:1fr; }
}
@media(prefers-reduced-motion:reduce) {
  *, *:before, *:after { animation:none !important; transition:none !important; }
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h3 {
  color: #F7F4FF !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] {
  margin-top: .35rem;
  margin-bottom: 1rem;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  min-height: 44px;
  border-radius: 10px !important;
}
.ri-hero-title { max-width: 900px; }
.ri-subtitle { max-width: 760px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* RetailIQ controls: consistent contrast for dropdowns and upload widgets */
[data-testid="stAppViewContainer"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #211a35 !important;
    border: 1px solid #8b5acb !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    min-height: 44px !important;
}
[data-testid="stAppViewContainer"] [data-baseweb="select"] input,
[data-testid="stAppViewContainer"] [data-baseweb="select"] span,
[data-testid="stAppViewContainer"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #ffffff !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background: #171326 !important;
    border: 1px solid #8b5acb !important;
    color: #f8f4ff !important;
}
[role="option"],
[data-baseweb="menu"] li {
    background: #171326 !important;
    color: #f8f4ff !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"],
[data-baseweb="menu"] li:hover {
    background: #482477 !important;
    color: #ffffff !important;
}
[data-testid="stFileUploader"] section {
    background: linear-gradient(135deg, #1b1630, #211a38) !important;
    border: 1px dashed #a66aff !important;
    border-radius: 16px !important;
    color: #f8f4ff !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section div,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section label {
    color: #eee6ff !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button[kind="secondary"],
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
    background: linear-gradient(110deg, #7835d5, #a94fff) !important;
    color: #ffffff !important;
    border: 1px solid #bd8aff !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {
    background: linear-gradient(110deg, #8d45ed, #c16aff) !important;
    color: #ffffff !important;
    box-shadow: 0 0 18px rgba(164, 88, 255, .38) !important;
}
[data-testid="stFileUploader"] svg {
    color: #d8b4fe !important;
    fill: #d8b4fe !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: #211a35 !important;
    border: 1px solid rgba(177,119,255,.3) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="ri-brand">RetailIQ <span>AI</span></div>', unsafe_allow_html=True)
st.caption("Smart Retail Analytics · Business Decision Support")

st.sidebar.markdown('<div class="ri-brand">RetailIQ <span>AI</span></div>', unsafe_allow_html=True)
st.sidebar.caption("SMART RETAIL ANALYTICS")

# CURRENCY CONFIGURATION

currency_options = {
    "INR": {"name": "Indian Rupee (₹)", "symbol": "₹"},
    "USD": {"name": "US Dollar ($)", "symbol": "$"},
    "GBP": {"name": "British Pound (£)", "symbol": "£"},
    "EUR": {"name": "Euro (€)", "symbol": "€"},
    "AED": {"name": "UAE Dirham (د.إ)", "symbol": "د.إ"},
    "CAD": {"name": "Canadian Dollar (C$)", "symbol": "C$"},
    "AUD": {"name": "Australian Dollar (A$)", "symbol": "A$"}
}

st.markdown("""
<style>
/* Dedicated fix for the currency selectbox in the sidebar.
   Streamlit/BaseWeb nests the selected value in several div/span layers,
   so target the widget specifically and override legacy sidebar colors. */
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
    background-color: #211a35 !important;
    background: #211a35 !important;
    border-color: #a66aff !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: #d8b4fe !important;
    fill: #d8b4fe !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    min-height: 44px !important;
    border-radius: 10px !important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="listbox"] {
    background-color: #171326 !important;
    color: #ffffff !important;
    border: 1px solid #8b5acb !important;
}
[data-baseweb="popover"] *,
[data-baseweb="menu"] *,
[role="listbox"] *,
[role="option"] {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"],
[data-baseweb="menu"] li:hover {
    background-color: #482477 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

selected_currency = st.sidebar.selectbox(
    "Display Currency (symbol only)",
    options=list(currency_options.keys()),
    format_func=lambda x: currency_options[x]["name"]
)

currency_symbol = currency_options[selected_currency]["symbol"]

st.markdown('<div class="ri-kicker">✦ AI RETAIL ANALYTICS PLATFORM</div>', unsafe_allow_html=True)
upload_col, hint_col = st.columns([1.35, 1])
with upload_col:
    st.markdown('<div class="ri-panel-title">Upload your business data</div><div class="ri-muted">CSV or Excel · Single file or multiple related files</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop files here or browse",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        help="Upload a retail or e-commerce transaction dataset. Related files can be uploaded together when they share business keys.",
        label_visibility="visible"
    )
with hint_col:
    st.markdown(
        '<div class="ri-glass" style="padding:1rem 1.1rem;margin-top:.2rem;">'
        '<div class="ri-panel-title">Ready for your data</div>'
        '<div class="ri-muted">RetailIQ profiles your data, detects useful business fields and prepares it for analysis.</div>'
        '<div style="margin-top:.65rem;color:#67e8c1;font-weight:700;font-size:.8rem;">● CSV &nbsp; ● XLSX &nbsp; ● Multiple related files</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ==========================================
# DATA LOADING & OLIST MULTI-FILE INTEGRATION
# ==========================================

import io
import os

@st.cache_data
def load_data(file_bytes, file_name):
    """Read CSVs with common encodings; leave Excel handling unchanged."""
    if file_name.lower().endswith(".csv"):
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, low_memory=False)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError(
            "Could not decode this CSV using UTF-8, UTF-8-BOM, Windows-1252 or Latin-1. "
            f"Please check that the file is a text CSV. Details: {last_error}"
        )
    return pd.read_excel(io.BytesIO(file_bytes))


def normalize_column(col):
    return "".join(ch.lower() for ch in str(col) if ch.isalnum())


def find_uploaded_file(file_map, required_name):
    """Find an uploaded CSV by its base filename, ignoring extension/case."""
    target = normalize_column(required_name)
    for normalized_name, dataframe in file_map.items():
        if normalized_name == target:
            return dataframe
    return None


if not uploaded_files:
    hero_left, hero_right = st.columns([1.25, 0.9], gap="large")
    with hero_left:
        st.markdown(
            '<div class="ri-kicker">✦ AI-POWERED RETAIL ANALYTICS PLATFORM</div>'
            '<div class="ri-hero-title">Turn Your Retail Data Into<br><span>Actionable Business Insights</span></div>'
            '<div class="ri-subtitle">A unified workspace to understand performance, explore sales trends, '
            'analyse customers and products, forecast demand, and make confident business decisions — '
            'using your own retail dataset.</div>',
            unsafe_allow_html=True
        )
        st.write("")
        st.markdown(
            '<div class="ri-muted">Built for retail and e-commerce teams that want to move from raw data '
            'to clear, practical decisions.</div>',
            unsafe_allow_html=True
        )
    with hero_right:
        st.markdown('<div class="ri-orb-wrap"><div class="ri-orb"></div></div>', unsafe_allow_html=True)

    st.markdown("### Explore what RetailIQ can do")
    module_data = [
        ("▥", "Business Overview", "See your key KPIs, revenue, orders and overall business health."),
        ("↗", "Sales Analytics", "Track sales trends, compare periods and understand what drives revenue."),
        ("⬡", "Product Intelligence", "Find leading products, category performance and demand patterns."),
        ("⌖", "Market Analysis", "Compare regions, customer segments and market contribution."),
        ("⌁", "Sales Forecast", "Use historical patterns to estimate future sales and plan ahead."),
        ("✧", "AI Business Advisor", "Turn detected business signals into prioritised actions and recommendations.")
    ]
    module_cols = st.columns(3)
    for idx, (icon_text, title_text, copy_text) in enumerate(module_data):
        with module_cols[idx % 3]:
            st.markdown(
                f'<div class="ri-glass ri-module"><div class="ri-module-icon">{icon_text}</div>'
                f'<div class="ri-module-title">{title_text}</div><div class="ri-module-copy">{copy_text}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("")
    feature_col, format_col = st.columns([1.2, 0.8], gap="large")
    with feature_col:
        st.markdown(
            '<div class="ri-info-panel"><div class="ri-info-title">✦ What happens when you upload data?</div>'
            '<div class="ri-feature-list">'
            '<div class="ri-feature"><span class="ri-feature-icon">✓</span><span><b>Automatic profiling</b><br>Understand rows, columns, missing values and data types.</span></div>'
            '<div class="ri-feature"><span class="ri-feature-icon">⌕</span><span><b>Smart field detection</b><br>Identify dates, revenue, products, customers and other useful fields.</span></div>'
            '<div class="ri-feature"><span class="ri-feature-icon">▤</span><span><b>Interactive analytics</b><br>Explore KPIs, charts, segments and business performance.</span></div>'
            '<div class="ri-feature"><span class="ri-feature-icon">↗</span><span><b>Predictive analysis</b><br>Review sales forecasts and customer behaviour models where data allows.</span></div>'
            '<div class="ri-feature"><span class="ri-feature-icon">✧</span><span><b>Business recommendations</b><br>See evidence-backed insights and practical next steps.</span></div>'
            '<div class="ri-feature"><span class="ri-feature-icon">⇩</span><span><b>Export your findings</b><br>Download analysis tables and reports for further use.</span></div>'
            '</div></div>',
            unsafe_allow_html=True
        )
    with format_col:
        st.markdown(
            '<div class="ri-info-panel"><div class="ri-info-title">Supported data formats</div>'
            '<div class="ri-feature"><span class="ri-feature-icon">CSV</span><span><b>CSV files</b><br>Transaction and structured tabular data.</span></div>'
            '<div class="ri-feature"><span class="ri-feature-icon">XLS</span><span><b>Excel files</b><br>Spreadsheet-based business data.</span></div>'
            '<div class="ri-feature"><span class="ri-feature-icon">＋</span><span><b>Multiple related files</b><br>Upload tables together when they connect through common identifiers.</span></div>'
            '<hr style="border-color:rgba(177,119,255,.18);margin:1rem 0;">'
            '<div class="ri-muted">Best results come from clear column headers, consistent dates and meaningful transaction records. '
            'Available analysis depends on the fields and data quality in your upload.</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("### How it works")
    step_cols = st.columns(3)
    steps = [
        ("1", "Upload your data", "Add a CSV, Excel file or related business tables."),
        ("2", "Prepare automatically", "RetailIQ profiles the data and detects useful fields."),
        ("3", "Explore insights", "Open dashboards, forecasts and recommendations.")
    ]
    for col, (number, title, copy) in zip(step_cols, steps):
        with col:
            st.markdown(
                f'<div class="ri-step"><div class="ri-step-number">{number}</div>'
                f'<div><div class="ri-step-title">{title}</div><div class="ri-step-copy">{copy}</div></div></div>',
                unsafe_allow_html=True
            )
    st.caption("Upload your dataset above to get started. RetailIQ adapts its analysis to the columns and information available.")
    st.stop()


try:
    uploaded_data = {}
    for file in uploaded_files:
        base_name = os.path.splitext(file.name)[0]
        file_bytes = file.getvalue()
        uploaded_data[normalize_column(base_name)] = load_data(
            file_bytes, file.name
        )
except Exception as e:
    st.error(f"Unable to read one of the uploaded files: {e}")
    st.stop()

# Batch 11: compact dataset context in the sidebar.
uploaded_table_rows = sum(len(table) for table in uploaded_data.values())
uploaded_table_columns = sum(len(table.columns) for table in uploaded_data.values())
st.sidebar.divider()
st.sidebar.markdown("**Current upload**")
sidebar_metrics = st.sidebar.columns(2)
sidebar_metrics[0].metric("Tables", f"{len(uploaded_data):,}")
sidebar_metrics[1].metric("Rows", f"{uploaded_table_rows:,}")
with st.sidebar.expander("Uploaded file details", expanded=False):
    for table_name, table_frame in uploaded_data.items():
        st.markdown(f"**{table_name}**")
        st.caption(f"{len(table_frame):,} rows · {len(table_frame.columns):,} columns")
    st.caption(f"Total columns across uploaded tables: {uploaded_table_columns:,}")

# Batch 12: deployment and upload readiness checks.
import platform
import streamlit as _streamlit_package

health_rows = []
for health_name, health_frame in uploaded_data.items():
    issues = []
    if health_frame.empty:
        issues.append("empty table")
    duplicate_column_count = int(health_frame.columns.duplicated().sum())
    if duplicate_column_count:
        issues.append(f"{duplicate_column_count} duplicate column label(s)")
    if len(health_frame.columns) == 0:
        issues.append("no columns detected")
    health_rows.append({
        "Uploaded table": health_name,
        "Rows": int(len(health_frame)),
        "Columns": int(len(health_frame.columns)),
        "Duplicate column labels": duplicate_column_count,
        "Readiness": "Review" if issues else "Ready",
        "Notes": "; ".join(issues) if issues else "Loaded successfully"
    })

health_report = pd.DataFrame(health_rows)
with st.sidebar.expander("App & data readiness", expanded=False):
    st.caption(f"Streamlit: {_streamlit_package.__version__}")
    st.caption(f"Python: {platform.python_version()}")
    st.caption(f"Pandas: {pd.__version__}")
    if not health_report.empty:
        ready_count = int((health_report["Readiness"] == "Ready").sum())
        st.metric("Tables ready", f"{ready_count}/{len(health_report)}")
        for _, health_row in health_report.iterrows():
            if health_row["Readiness"] == "Ready":
                st.success(f"{health_row['Uploaded table']}: ready")
            else:
                st.warning(f"{health_row['Uploaded table']}: {health_row['Notes']}")
        st.download_button(
            "Download Upload Health Report (CSV)",
            data=health_report.to_csv(index=False),
            file_name="retailiq_upload_health_report.csv",
            mime="text/csv",
            key="retailiq_upload_health_report_download"
        )
    else:
        st.info("Upload a dataset to run readiness checks.")



# ==========================================
# ADAPTIVE DATA PROFILING & GENERIC ANALYTICS
# ==========================================

import re


@st.cache_data(show_spinner=False)
def _cached_csv_export(frame):
    """Cache CSV serialization for unchanged analysis tables."""
    return frame.to_csv(index=False)


@st.cache_data(show_spinner=False)
def _cached_visual_sample(frame, columns, sample_size=50000):
    """Return a deterministic sample for charts when datasets are very large."""
    selected = frame.loc[:, list(columns)]
    if len(selected) <= sample_size:
        return selected.copy()
    return selected.sample(n=sample_size, random_state=42)


@st.cache_data(show_spinner=False)
def _cached_numeric_correlation(frame, columns, sample_size=50000):
    """Calculate correlation on all rows for small data and a stable sample for large data."""
    sample = _cached_visual_sample(frame, list(columns), sample_size)
    numeric = sample.apply(pd.to_numeric, errors="coerce")
    return numeric.corr()


@st.cache_data(show_spinner=False)
def _cached_monthly_trend(frame, date_col, measure, aggregation):
    """Aggregate the full dataset to monthly values and cache the result."""
    temp = frame[[date_col, measure]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[measure] = pd.to_numeric(temp[measure], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        return pd.DataFrame(columns=[date_col, measure])
    grouped_series = temp.set_index(date_col)[measure].resample("MS")
    if aggregation == "Sum":
        result = grouped_series.sum()
    elif aggregation == "Average":
        result = grouped_series.mean()
    else:
        result = grouped_series.count()
    return result.reset_index(name=measure)

def _clean_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _is_identifier_name(column):
    """Identify record keys and code-like columns without treating all numbers as measures."""
    name = _clean_name(column)
    exact_ids = {
        "id", "rownumber", "rownumberid", "index", "recordnumber", "serialnumber",
        "customerid", "customeruniqueid", "employeeid", "employeenumber", "productid", "orderid",
        "orderitemid", "invoiceid", "invoiceno", "transactionid", "sellerid",
        "departmentid", "categoryid", "stockcode", "productcode", "sku", "uuid",
        "zipcode", "postalcode", "phonenumber", "accesskey"
    }
    return (
        name in exact_ids
        or name.endswith("id")
        or name.endswith("uuid")
        or name.endswith("code")
        or name.endswith("key")
        or name in {"latitude", "longitude", "lat", "lon", "lng"}
        or name in {"customeremail", "customerpassword", "customerstreet"}
    )


def _is_flag_name(column):
    name = _clean_name(column)
    flag_tokens = (
        "attrition", "exited", "churn", "isactive", "hascrcard", "late_delivery_risk",
        "late delivery risk", "flag", "risk", "isdelivered", "isreturned", "cancelled"
    )
    return any(_clean_name(token) in name for token in flag_tokens)


@st.cache_data(show_spinner=False)
def _profile_table(frame):
    profile = {
        "rows": len(frame), "columns": len(frame.columns), "numeric": [],
        "dates": [], "categorical": [], "identifiers": [], "geo": [], "metadata": [],
        "missing": {}, "unique": {}, "duplicate_rows": int(frame.duplicated().sum())
    }
    for col in frame.columns:
        series = frame[col]
        profile["missing"][col] = int(series.isna().sum())
        profile["unique"][col] = int(series.nunique(dropna=True))
        name = _clean_name(col)

        if name in {"latitude", "longitude", "lat", "lon", "lng"}:
            profile["geo"].append(col)
            continue
        if _is_identifier_name(col):
            profile["identifiers"].append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            profile["dates"].append(col)
            continue

        date_name = any(token in name for token in (
            "date", "time", "timestamp", "created", "updated", "month", "year"
        ))
        if date_name and not pd.api.types.is_numeric_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
            nonmissing = int(series.notna().sum())
            if nonmissing and parsed.notna().sum() / nonmissing >= 0.65:
                profile["dates"].append(col)
                continue

        converted = pd.to_numeric(series, errors="coerce")
        nonmissing = int(series.notna().sum())
        numeric_ratio = converted.notna().sum() / nonmissing if nonmissing else 0
        unique_count = profile["unique"][col]
        if unique_count <= 1 and pd.api.types.is_numeric_dtype(series):
            profile["metadata"].append(col)
            continue
        if _is_flag_name(col) or (unique_count == 2 and numeric_ratio >= 0.85):
            profile["categorical"].append(col)
        elif pd.api.types.is_numeric_dtype(series) or numeric_ratio >= 0.85:
            profile["numeric"].append(col)
        elif unique_count <= max(50, int(len(series) * 0.20)):
            profile["categorical"].append(col)

    return profile


def _retail_schema_is_compatible(frame):
    aliases = {
        "order": {"invoiceno", "invoice", "invoiceid", "orderid", "ordernumber", "transactionid"},
        "product": {"stockcode", "productid", "productcode", "sku", "itemcode"},
        "date": {"invoicedate", "orderdate", "transactiondate", "date", "orderpurchasetimestamp"},
        "quantity": {"quantity", "qty", "unitssold", "numberofitems"},
        "price": {"unitprice", "sellingprice", "priceperunit", "price"},
    }
    normalized = {_clean_name(c) for c in frame.columns}
    return all(bool(normalized & values) for values in aliases.values())


def _find_column(frame, aliases):
    lookup = {_clean_name(c): c for c in frame.columns}
    for alias in aliases:
        if _clean_name(alias) in lookup:
            return lookup[_clean_name(alias)]
    return None


@st.cache_data(show_spinner=False)
def _semantic_column_mapping(frame):
    """Suggest business roles from common column-name synonyms, with type checks."""
    role_aliases = {
        "Revenue / Sales": ["revenue", "sales", "netsales", "totalsales", "salesamount",
                            "revenueamount", "orderamount", "totalamount", "amountpaid",
                            "netamount", "grosssales", "orderitemtotal"],
        "Unit price": ["unitprice", "priceperunit", "sellingprice", "saleprice",
                       "itemprice", "productprice", "price"],
        "Quantity / Units": ["quantity", "qty", "unitssold", "itemsquantity",
                             "numberofitems", "orderquantity", "units"],
        "Order / Transaction ID": ["orderid", "orderno", "ordernumber", "invoiceno",
                                   "invoiceid", "transactionid", "purchaseid"],
        "Product": ["productname", "product", "productid", "sku", "stockcode",
                    "itemname", "itemcode", "productcode"],
        "Customer": ["customerid", "customeruniqueid", "customername", "clientid",
                     "buyerid", "userid"],
        "Date / Timestamp": ["orderdate", "invoicedate", "transactiondate",
                             "purchasedate", "createdat", "timestamp", "date"],
        "Geography": ["country", "region", "state", "city", "market", "geography",
                      "province", "postalcode", "zipcode"],
        "Category": ["category", "productcategory", "department", "segment",
                     "producttype", "itemcategory"],
        "Status": ["orderstatus", "deliverystatus", "paymentstatus", "status",
                   "shippingstatus"],
        "Rating / Review": ["reviewscore", "rating", "customerrating",
                            "feedbackscore", "satisfaction", "stars"],
        "Shipping / Delivery time": ["shippingdays", "deliverydays", "actualshippingdays",
                                     "daysforshipping", "leadtime", "transittime"],
        "Churn / Attrition": ["churn", "churned", "ischurn", "exited",
                              "attrition", "employeeattrition", "leftcompany"],
    }
    numeric_roles = {"Revenue / Sales", "Unit price", "Quantity / Units",
                     "Rating / Review", "Shipping / Delivery time"}
    mappings = []
    normalized_columns = {col: _clean_name(col) for col in frame.columns}
    for role, aliases in role_aliases.items():
        candidates = []
        for col, normalized in normalized_columns.items():
            if normalized in aliases:
                score, reason = 1.0, "Exact synonym"
            elif any(len(alias) >= 5 and (normalized.startswith(alias) or normalized.endswith(alias))
                     for alias in aliases):
                score, reason = 0.82, "Name variation"
            else:
                continue
            if role in numeric_roles:
                converted = pd.to_numeric(frame[col], errors="coerce")
                nonmissing = int(frame[col].notna().sum())
                ratio = converted.notna().sum() / nonmissing if nonmissing else 0
                if ratio < 0.70:
                    continue
            if role == "Date / Timestamp":
                parsed = pd.to_datetime(frame[col], errors="coerce")
                nonmissing = int(frame[col].notna().sum())
                ratio = parsed.notna().sum() / nonmissing if nonmissing else 0
                if ratio < 0.65:
                    continue
            candidates.append((score, col, reason))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if candidates:
            best_score, best_col, reason = candidates[0]
            ambiguous = len(candidates) > 1 and candidates[1][0] == best_score
            mappings.append({
                "Business role": role,
                "Suggested column": str(best_col),
                "Confidence": "Review — multiple matches" if ambiguous else
                              ("High" if best_score >= 1 else "Medium"),
                "Match basis": reason
            })
        else:
            mappings.append({
                "Business role": role, "Suggested column": "Not detected",
                "Confidence": "Unavailable", "Match basis": "No validated synonym found"
            })
    return mappings


@st.cache_data(show_spinner=False)
def _generate_business_insights(frame, roles):
    """Generate deterministic observations only when supporting columns/data exist."""
    insights = []
    measure_col = roles.get("Revenue / Sales") or roles.get("Quantity / Units")
    category_col = roles.get("Category") or roles.get("Product") or roles.get("Geography")
    date_col = roles.get("Date / Timestamp")
    status_col = roles.get("Status")
    rating_col = roles.get("Rating / Review")

    if measure_col:
        values = pd.to_numeric(frame[measure_col], errors="coerce")
        valid = values.dropna()
        if not valid.empty:
            insights.append(
                ("Measure summary",
                 f"{measure_col} totals {valid.sum():,.2f} across {len(valid):,} valid records; "
                 f"the average per record is {valid.mean():,.2f}.")
            )
            if category_col:
                temp = pd.DataFrame({
                    "category": frame[category_col].astype("string"),
                    "value": values
                }).dropna()
                grouped = temp.groupby("category")["value"].sum().sort_values(ascending=False)
                if not grouped.empty:
                    top_name, top_value = str(grouped.index[0]), float(grouped.iloc[0])
                    share = top_value / grouped.sum() * 100 if grouped.sum() else None
                    share_text = f" ({share:.1f}% of the grouped total)" if share is not None else ""
                    insights.append(
                        ("Top contributing group",
                         f"{top_name} has the highest total {measure_col}: {top_value:,.2f}{share_text}.")
                    )
                    if len(grouped) >= 2:
                        low_name, low_value = str(grouped.index[-1]), float(grouped.iloc[-1])
                        insights.append(
                            ("Lowest contributing group",
                             f"{low_name} has the lowest total among the displayed groups: {low_value:,.2f}.")
                        )

            if date_col:
                dates = pd.to_datetime(frame[date_col], errors="coerce")
                trend_data = pd.DataFrame({"date": dates, "value": values}).dropna()
                if not trend_data.empty:
                    monthly = trend_data.set_index("date")["value"].resample("MS").sum()
                    monthly = monthly[monthly.notna()]
                    if len(monthly) >= 2:
                        first, last = float(monthly.iloc[0]), float(monthly.iloc[-1])
                        if first != 0:
                            change = (last - first) / abs(first) * 100
                            direction = "increased" if change > 0 else "decreased" if change < 0 else "remained unchanged"
                            insights.append(
                                ("Monthly movement",
                                 f"Monthly {measure_col} {direction} by {abs(change):.1f}% "
                                 f"between {monthly.index[0].strftime('%b %Y')} and "
                                 f"{monthly.index[-1].strftime('%b %Y')}.")
                            )
                        else:
                            insights.append(
                                ("Monthly movement",
                                 f"The first month has a zero total, so a percentage change cannot be calculated. "
                                 f"The latest month total is {last:,.2f}.")
                            )

    if status_col:
        counts = frame[status_col].fillna("Missing").astype(str).value_counts()
        if len(counts):
            label, count = counts.index[0], int(counts.iloc[0])
            insights.append(
                ("Most common status",
                 f"{label} is the most frequent {status_col} value ({count:,} records).")
            )
    if rating_col:
        ratings = pd.to_numeric(frame[rating_col], errors="coerce").dropna()
        if not ratings.empty:
            insights.append(
                ("Customer rating",
                 f"Average {rating_col} is {ratings.mean():.2f} across {len(ratings):,} valid ratings.")
            )
    return insights



@st.cache_data(show_spinner=False)
def _generate_business_recommendations(frame, roles):
    """Create transparent, rule-based actions from validated business roles."""
    recommendations = []
    measure_col = roles.get("Revenue / Sales") or roles.get("Quantity / Units")
    category_col = roles.get("Category") or roles.get("Product") or roles.get("Geography")
    date_col = roles.get("Date / Timestamp")
    rating_col = roles.get("Rating / Review")
    shipping_col = roles.get("Shipping / Delivery time")
    churn_col = roles.get("Churn / Attrition")

    if measure_col and category_col:
        values = pd.to_numeric(frame[measure_col], errors="coerce")
        grouped = pd.DataFrame({"group": frame[category_col].astype("string"), "value": values}).dropna()
        totals = grouped.groupby("group")["value"].sum().sort_values(ascending=False)
        grand_total = float(totals.sum()) if not totals.empty else 0.0
        if len(totals) >= 2 and grand_total > 0:
            share = float(totals.iloc[0]) / grand_total * 100
            if share >= 60:
                recommendations.append(("High", "Reduce dependence on the leading group",
                    f"{totals.index[0]} contributes {share:.1f}% of grouped {measure_col}. Review concentration risk and test ways to grow other groups."))
            elif share >= 40:
                recommendations.append(("Medium", "Monitor group concentration",
                    f"{totals.index[0]} contributes {share:.1f}% of grouped {measure_col}. Track its performance alongside the next-largest groups."))

    if measure_col and date_col:
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        vals = pd.to_numeric(frame[measure_col], errors="coerce")
        trend = pd.DataFrame({"date": dates, "value": vals}).dropna()
        if not trend.empty:
            monthly = trend.set_index("date")["value"].resample("MS").sum()
            if len(monthly) >= 2 and float(monthly.iloc[0]) != 0:
                change = (float(monthly.iloc[-1]) - float(monthly.iloc[0])) / abs(float(monthly.iloc[0])) * 100
                if change <= -10:
                    recommendations.append(("High", "Investigate the downward trend",
                        f"{measure_col} is down {abs(change):.1f}% from the first to the latest month in the data. Compare product, market, pricing and campaign segments to identify possible drivers."))
                elif change >= 10:
                    recommendations.append(("Medium", "Understand and sustain recent growth",
                        f"{measure_col} is up {change:.1f}% from the first to the latest month in the data. Identify the groups contributing most and check inventory or operational capacity."))

    if rating_col:
        ratings = pd.to_numeric(frame[rating_col], errors="coerce").dropna()
        if len(ratings) and ratings.mean() <= 3.0:
            recommendations.append(("High", "Review customer experience signals",
                f"Average {rating_col} is {ratings.mean():.2f}. Segment low scores by product, category or market and review feedback before choosing corrective actions."))

    if shipping_col:
        shipping = pd.to_numeric(frame[shipping_col], errors="coerce").dropna()
        if len(shipping) and shipping.mean() > 7:
            recommendations.append(("Medium", "Review delivery lead times",
                f"Average {shipping_col} is {shipping.mean():.1f}. Compare delivery times by carrier, region or product where those fields are available."))

    if churn_col:
        raw = frame[churn_col].dropna()
        normalized = raw.astype(str).str.strip().str.lower()
        positive = normalized.isin(["1", "true", "yes", "y", "churn", "churned", "exited", "left"])
        negative = normalized.isin(["0", "false", "no", "n", "active", "retained", "stayed"])
        valid = positive | negative
        if valid.any():
            rate = positive[valid].mean() * 100
            if rate >= 20:
                recommendations.append(("High", "Examine churn or attrition segments",
                    f"The detected positive churn/attrition rate is {rate:.1f}% among recognizable records. Compare cohorts and customer or employee segments; this rate alone does not explain why people leave."))

    missing_pct = float(frame.isna().sum().sum()) / max(1, frame.size) * 100
    if missing_pct >= 10:
        recommendations.append(("Medium", "Improve data completeness",
            f"{missing_pct:.1f}% of cells are missing. Prioritize fields used in key metrics and investigate the source of missing records before making decisions."))

    if not recommendations:
        recommendations.append(("Info", "Build a baseline before acting",
            "No strong rule-based action was triggered by the detected fields. Review the column mapping and collect enough date, category and outcome data to support deeper analysis."))
    return recommendations

@st.cache_data(show_spinner=False)
def _build_data_quality_report(frame, profile):
    """Create non-destructive data-quality diagnostics; never alter source rows."""
    rows = []
    row_count = max(1, len(frame))
    for col in frame.columns:
        series = frame[col]
        missing = int(series.isna().sum())
        nonmissing = int(series.notna().sum())
        name = _clean_name(col)
        issue_count = 0
        issue_label = "Missing values"
        if col in profile.get("dates", []):
            parsed = pd.to_datetime(series, errors="coerce")
            issue_count = int((series.notna() & parsed.isna()).sum())
            issue_label = "Unparseable dates"
        elif col in profile.get("numeric", []):
            converted = pd.to_numeric(series, errors="coerce")
            issue_count = int((series.notna() & converted.isna()).sum())
            issue_label = "Non-numeric values"
        elif pd.api.types.is_numeric_dtype(series) and nonmissing >= 8:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) >= 8 and numeric.nunique() > 3:
                q1, q3 = numeric.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    issue_count = int(((numeric < q1 - 1.5 * iqr) |
                                       (numeric > q3 + 1.5 * iqr)).sum())
                    issue_label = "Potential outliers (IQR)"
        rows.append({
            "Column": str(col),
            "Missing values": missing,
            "Missing %": round(missing / row_count * 100, 2),
            "Additional check": issue_label,
            "Additional issues": issue_count
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _recommend_chart_types(frame, profile, roles):
    """Recommend chart forms from detected field roles and cardinality."""
    recommendations = []
    date_col = roles.get("Date / Timestamp")
    category_col = roles.get("Category") or roles.get("Geography") or roles.get("Product")
    measure_col = roles.get("Revenue / Sales") or roles.get("Quantity / Units")
    numeric_cols = profile.get("numeric", [])

    if date_col and measure_col:
        recommendations.append(("Time trend", "Line chart", f"{measure_col} over {date_col}"))
    if category_col:
        count = frame[category_col].nunique(dropna=True)
        chart = "Donut chart" if 2 <= count <= 5 else "Horizontal bar chart"
        detail = f"Compare {category_col} across {count:,} distinct values"
        if measure_col:
            detail += f" using {measure_col}"
        recommendations.append(("Category comparison", chart, detail))
    if measure_col:
        recommendations.append(("Measure distribution", "Histogram", f"Understand the spread of {measure_col}"))
    if len(numeric_cols) >= 2:
        recommendations.append(("Measure relationship", "Scatter plot / correlation", "Explore association between numeric measures"))
    return recommendations



@st.cache_data(show_spinner=False)
def _calculate_rfm_analysis(frame, roles):
    """Build customer-level Recency, Frequency and Monetary metrics from validated fields."""
    customer_col = roles.get("Customer")
    date_col = roles.get("Date / Timestamp")
    revenue_col = roles.get("Revenue / Sales")
    order_col = roles.get("Order / Transaction ID")

    if not all([customer_col, date_col, revenue_col]):
        return pd.DataFrame(), "RFM needs validated Customer, Date and Revenue/Sales columns."

    columns = [customer_col, date_col, revenue_col]
    if order_col and order_col not in columns:
        columns.append(order_col)
    data = frame[columns].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[revenue_col] = pd.to_numeric(data[revenue_col], errors="coerce")
    data = data.dropna(subset=[customer_col, date_col, revenue_col])
    if data.empty:
        return pd.DataFrame(), "No valid customer, date and revenue records were available."

    data = data[data[revenue_col] >= 0]
    if data.empty:
        return pd.DataFrame(), "No non-negative revenue records were available for RFM analysis."

    snapshot_date = data[date_col].max().normalize() + pd.Timedelta(days=1)
    grouped = data.groupby(customer_col, dropna=False)
    rfm = grouped.agg(
        Last_purchase=(date_col, "max"),
        Monetary=(revenue_col, "sum"),
        Transactions=(date_col, "size"),
    ).reset_index()
    if order_col:
        frequency = data.groupby(customer_col, dropna=False)[order_col].nunique(dropna=True)
        rfm["Frequency"] = rfm[customer_col].map(frequency).fillna(0).astype(int)
        frequency_note = f"Frequency counts distinct values in '{order_col}'."
    else:
        rfm["Frequency"] = rfm["Transactions"]
        frequency_note = "No validated order ID was found; frequency uses transaction rows as a proxy."

    rfm["Recency (days)"] = (snapshot_date - rfm["Last_purchase"]).dt.days
    rfm["Monetary"] = rfm["Monetary"].astype(float)
    if len(rfm) >= 2:
        rfm["R_score"] = pd.qcut(
            rfm["Recency (days)"].rank(method="first", ascending=False),
            5, labels=[5, 4, 3, 2, 1]
        ).astype(int)
        rfm["F_score"] = pd.qcut(
            rfm["Frequency"].rank(method="first"),
            5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        rfm["M_score"] = pd.qcut(
            rfm["Monetary"].rank(method="first"),
            5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
    else:
        rfm["R_score"] = 3
        rfm["F_score"] = 3
        rfm["M_score"] = 3

    rfm["RFM_score"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)
    rfm["Segment"] = np.select(
        [
            rfm["RFM_score"] >= 13,
            rfm["RFM_score"].between(10, 12),
            rfm["RFM_score"].between(8, 9),
            rfm["RFM_score"].between(6, 7),
        ],
        ["Champions", "Loyal customers", "Potential loyalists", "At risk"],
        default="Hibernating"
    )
    return rfm, frequency_note


@st.cache_data(show_spinner=False)
def _calculate_customer_cohorts(frame, roles):
    """Calculate monthly customer cohorts and their subsequent active-month retention."""
    customer_col = roles.get("Customer")
    date_col = roles.get("Date / Timestamp")
    if not customer_col or not date_col:
        return pd.DataFrame(), "Cohorts need validated Customer and Date columns."

    data = frame[[customer_col, date_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[customer_col, date_col]).drop_duplicates()
    if data.empty:
        return pd.DataFrame(), "No valid customer-date records were available."

    data["Order month"] = data[date_col].dt.to_period("M")
    first_purchase = data.groupby(customer_col)["Order month"].min().rename("Cohort")
    data = data.join(first_purchase, on=customer_col)
    data["Cohort index"] = (
        (data["Order month"].dt.year - data["Cohort"].dt.year) * 12
        + data["Order month"].dt.month - data["Cohort"].dt.month + 1
    )
    cohort_counts = data.groupby(["Cohort", "Cohort index"])[customer_col].nunique().unstack(fill_value=0)
    if cohort_counts.empty:
        return pd.DataFrame(), "No monthly customer cohorts could be calculated."
    cohort_sizes = cohort_counts.iloc[:, 0].replace(0, np.nan)
    retention = cohort_counts.divide(cohort_sizes, axis=0) * 100
    retention.index = retention.index.astype(str)
    retention.columns = [f"Month {int(col)}" for col in retention.columns]
    return retention.round(1), "Retention is the share of each first-purchase cohort active in each subsequent calendar month."


@st.cache_data(show_spinner=False)
def _calculate_advanced_customer_intelligence(frame, roles):
    """Calculate historical customer value, repeat-purchase and revenue-concentration metrics."""
    customer_col = roles.get("Customer")
    revenue_col = roles.get("Revenue / Sales")
    date_col = roles.get("Date / Timestamp")
    order_col = roles.get("Order / Transaction ID")
    if not customer_col or not revenue_col:
        return None, "Advanced customer intelligence needs validated Customer and Revenue/Sales fields."

    cols = [customer_col, revenue_col]
    if date_col:
        cols.append(date_col)
    if order_col and order_col not in cols:
        cols.append(order_col)
    data = frame[cols].copy()
    data[revenue_col] = pd.to_numeric(data[revenue_col], errors="coerce")
    data = data.dropna(subset=[customer_col, revenue_col])
    data = data[data[revenue_col] >= 0]
    if data.empty:
        return None, "No valid non-negative customer revenue records were available."

    agg = {
        "Historical revenue": (revenue_col, "sum"),
        "Transaction rows": (revenue_col, "size"),
    }
    if date_col:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        agg["First purchase"] = (date_col, "min")
        agg["Last purchase"] = (date_col, "max")
    if order_col:
        agg["Distinct orders"] = (order_col, "nunique")
    customers = data.groupby(customer_col, dropna=False).agg(**agg).reset_index()
    frequency_col = "Distinct orders" if order_col else "Transaction rows"
    customers["Purchase frequency"] = customers[frequency_col].fillna(0).astype(int)
    customers["Customer type"] = np.where(
        customers["Purchase frequency"] >= 2, "Repeat customer", "One-time customer"
    )
    customers = customers.sort_values("Historical revenue", ascending=False).reset_index(drop=True)
    total_revenue = float(customers["Historical revenue"].sum())
    customer_count = len(customers)
    repeat_count = int((customers["Customer type"] == "Repeat customer").sum())
    customers["Revenue share (%)"] = (
        customers["Historical revenue"] / total_revenue * 100 if total_revenue > 0 else 0
    )
    customers["Cumulative revenue share (%)"] = customers["Revenue share (%)"].cumsum()
    top_n = max(1, int(np.ceil(customer_count * 0.20)))
    top_customer_share = (
        float(customers.head(top_n)["Historical revenue"].sum() / total_revenue * 100)
        if total_revenue > 0 else 0.0
    )
    order_count = (
        int(data[order_col].nunique()) if order_col else int(data.shape[0])
    )
    average_order_value = total_revenue / order_count if order_count else 0.0
    summary = {
        "customers": customers,
        "customer_count": customer_count,
        "repeat_customer_count": repeat_count,
        "repeat_rate": repeat_count / customer_count * 100 if customer_count else 0.0,
        "historical_revenue": total_revenue,
        "revenue_per_customer": total_revenue / customer_count if customer_count else 0.0,
        "average_order_value": average_order_value,
        "top_customer_count": top_n,
        "top_customer_revenue_share": top_customer_share,
        "frequency_note": (
            f"Purchase frequency uses distinct values in '{order_col}'."
            if order_col else
            "No validated order ID was found; transaction rows are used as a purchase-frequency proxy."
        ),
    }
    return summary, "Customer value metrics describe historical activity in the selected dataset; they are not forward-looking lifetime value predictions."


@st.cache_data(show_spinner=False)
def _train_churn_classification(frame, roles):
    """Train a transparent logistic classifier when a valid binary churn target exists."""
    target_col = roles.get("Churn / Attrition")
    if not target_col:
        return None, "No high-confidence churn/attrition target was detected."

    raw_target = frame[target_col]
    normalized = raw_target.astype(str).str.strip().str.lower()
    positive_values = {"1", "true", "yes", "y", "churn", "churned", "exited", "left"}
    negative_values = {"0", "false", "no", "n", "active", "retained", "stay", "stayed"}
    recognized = normalized.isin(positive_values | negative_values)
    data = frame.loc[recognized].copy()
    y = normalized.loc[recognized].isin(positive_values).astype(int).to_numpy()

    if len(data) < 40 or len(np.unique(y)) != 2:
        return None, "At least 40 records with both recognizable churn and non-churn outcomes are required."
    class_counts = np.bincount(y, minlength=2)
    if class_counts.min() < 8:
        return None, "At least 8 examples from each outcome class are required for a meaningful holdout evaluation."

    excluded = {
        target_col,
        roles.get("Customer"),
        roles.get("Order / Transaction ID"),
        roles.get("Date / Timestamp"),
    }
    candidate_cols = [
        col for col in data.columns
        if col not in excluded
        and not _is_identifier_name(col)
        and data[col].nunique(dropna=True) > 1
    ]
    feature_parts = []
    feature_names = []
    for col in candidate_cols:
        series = data[col]
        numeric = pd.to_numeric(series, errors="coerce")
        nonmissing = int(series.notna().sum())
        numeric_ratio = numeric.notna().sum() / max(1, nonmissing)
        if pd.api.types.is_numeric_dtype(series) or numeric_ratio >= 0.85:
            values = numeric.replace([np.inf, -np.inf], np.nan)
            if values.notna().sum() < 10:
                continue
            median = float(values.median())
            values = values.fillna(median).to_numpy(dtype=float)
            feature_parts.append(values.reshape(-1, 1))
            feature_names.append(str(col))
        else:
            unique_count = series.nunique(dropna=True)
            if 2 <= unique_count <= 12:
                filled = series.fillna("Missing").astype(str)
                dummies = pd.get_dummies(filled, prefix=str(col), dtype=float)
                if dummies.shape[1] > 12:
                    continue
                feature_parts.append(dummies.to_numpy(dtype=float))
                feature_names.extend(dummies.columns.astype(str).tolist())

    if not feature_parts:
        return None, "No suitable numeric or low-cardinality categorical predictor columns were found."
    X = np.concatenate(feature_parts, axis=1)
    if X.shape[1] > 60:
        return None, "More than 60 candidate features were detected. Reduce categorical complexity before modelling."
    if not np.isfinite(X).all():
        return None, "Some predictor values could not be prepared safely for modelling."

    # Stratified, deterministic holdout split.
    rng = np.random.default_rng(42)
    train_indices, test_indices = [], []
    for label in (0, 1):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        split_at = max(1, min(len(indices) - 1, int(len(indices) * 0.8)))
        train_indices.extend(indices[:split_at].tolist())
        test_indices.extend(indices[split_at:].tolist())
    train_indices = np.array(train_indices, dtype=int)
    test_indices = np.array(test_indices, dtype=int)
    if len(test_indices) < 4:
        return None, "The holdout set is too small to evaluate this model."

    x_train, x_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    scales[scales == 0] = 1.0
    x_train = np.clip((x_train - means) / scales, -8, 8)
    x_test = np.clip((x_test - means) / scales, -8, 8)
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])

    # Balanced logistic regression trained with gradient descent and L2 regularization.
    weights = np.zeros(x_train.shape[1], dtype=float)
    positive_weight = len(y_train) / (2 * max(1, int(y_train.sum())))
    negative_weight = len(y_train) / (2 * max(1, int((y_train == 0).sum())))
    sample_weights = np.where(y_train == 1, positive_weight, negative_weight)
    learning_rate = 0.08
    for _ in range(1200):
        logits = np.clip(x_train @ weights, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = (x_train.T @ ((probabilities - y_train) * sample_weights)) / len(y_train)
        gradient[1:] += 0.01 * weights[1:]
        weights -= learning_rate * gradient

    test_probabilities = 1.0 / (1.0 + np.exp(-np.clip(x_test @ weights, -30, 30)))
    predictions = (test_probabilities >= 0.5).astype(int)
    tp = int(((predictions == 1) & (y_test == 1)).sum())
    tn = int(((predictions == 0) & (y_test == 0)).sum())
    fp = int(((predictions == 1) & (y_test == 0)).sum())
    fn = int(((predictions == 0) & (y_test == 1)).sum())
    accuracy = (tp + tn) / max(1, len(y_test))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    # AUC by pairwise ranking; safe for the small holdout used here.
    positives = test_probabilities[y_test == 1]
    negatives = test_probabilities[y_test == 0]
    auc = float(np.mean([
        (pos > neg) + 0.5 * (pos == neg)
        for pos in positives for neg in negatives
    ])) if len(positives) and len(negatives) else float("nan")

    result = {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "auc": auc, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "features": feature_names, "records": len(data),
        "test_records": len(test_indices),
        "baseline_rate": float(y_test.mean()),
    }
    return result, "A deterministic logistic-regression model was evaluated on a stratified 20% holdout set. Metrics are estimates, not guarantees of future performance."


@st.cache_data(show_spinner=False)
def _evaluate_time_series_forecast(series):
    """Backtest an additive trend model on the last 20% of daily observations."""
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(values) < 35:
        return None, "At least 35 daily observations are required for a holdout backtest."
    holdout_size = max(7, int(round(len(values) * 0.2)))
    if len(values) - holdout_size < 28:
        holdout_size = len(values) - 28
    train, actual = values.iloc[:-holdout_size], values.iloc[-holdout_size:]
    if len(train) < 28 or len(actual) < 7:
        return None, "There are not enough observations for a reliable holdout evaluation."
    try:
        seasonal = len(train) >= 60
        fitted = ExponentialSmoothing(
            train, trend="add", seasonal="add" if seasonal else None,
            seasonal_periods=7 if seasonal else None, initialization_method="estimated"
        ).fit()
        predicted = np.asarray(fitted.forecast(len(actual)), dtype=float)
        observed = actual.to_numpy(dtype=float)
        errors = observed - predicted
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        nonzero = np.abs(observed) > 1e-9
        mape = float(np.mean(np.abs(errors[nonzero] / observed[nonzero])) * 100) if nonzero.any() else float("nan")
        return {
            "mae": mae, "rmse": rmse, "mape": mape,
            "actual": actual, "predicted": pd.Series(predicted, index=actual.index),
            "holdout_size": len(actual),
        }, "Backtest compares predictions with the most recent held-out observations; it does not guarantee future accuracy."
    except Exception as exc:
        return None, f"Backtest could not be completed: {exc}"


@st.cache_data(show_spinner=False)
def _forecast_from_daily_series(series, horizon=30):
    """Fit the existing additive Exponential Smoothing approach and forecast future periods."""
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(values) < 28:
        return pd.Series(dtype=float)
    seasonal = len(values) >= 60
    fitted = ExponentialSmoothing(
        values, trend="add", seasonal="add" if seasonal else None,
        seasonal_periods=7 if seasonal else None, initialization_method="estimated"
    ).fit()
    return fitted.forecast(horizon)


@st.cache_data(show_spinner=False)
def _build_executive_insight_summary(frame, roles, insights, recommendations):
    """Create a concise, evidence-linked executive summary using transparent rules."""
    summary_parts = []
    measure_col = roles.get("Revenue / Sales") or roles.get("Quantity / Units")
    date_col = roles.get("Date / Timestamp")
    category_col = roles.get("Category") or roles.get("Product") or roles.get("Geography")

    if measure_col and measure_col in frame.columns:
        values = pd.to_numeric(frame[measure_col], errors="coerce").dropna()
        if not values.empty:
            summary_parts.append(
                f"The selected dataset contains {len(frame):,} records. "
                f"Recorded {measure_col.lower()} totals {values.sum():,.2f}, "
                f"with an average of {values.mean():,.2f} per valid record."
            )
    if category_col and category_col in frame.columns and measure_col:
        values = pd.to_numeric(frame[measure_col], errors="coerce")
        grouped = pd.DataFrame({
            "group": frame[category_col].astype("string"),
            "value": values
        }).dropna().groupby("group")["value"].sum().sort_values(ascending=False)
        if not grouped.empty:
            total = grouped.sum()
            share = grouped.iloc[0] / total * 100 if total else 0
            summary_parts.append(
                f"{grouped.index[0]} is the largest contributing {category_col.lower()} "
                f"({share:.1f}% of grouped {measure_col.lower()})."
            )
    if date_col and measure_col:
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        values = pd.to_numeric(frame[measure_col], errors="coerce")
        trend = pd.DataFrame({"date": dates, "value": values}).dropna()
        if not trend.empty:
            monthly = trend.set_index("date")["value"].resample("MS").sum()
            if len(monthly) >= 2 and monthly.iloc[0] != 0:
                change = (monthly.iloc[-1] - monthly.iloc[0]) / abs(monthly.iloc[0]) * 100
                direction = "higher" if change > 0 else "lower" if change < 0 else "unchanged"
                summary_parts.append(
                    f"The latest observed month is {abs(change):.1f}% {direction} "
                    f"than the first observed month in the selected period."
                )

    if not summary_parts:
        summary_parts.append(
            f"The selected table contains {len(frame):,} records and {len(frame.columns):,} columns. "
            "Validated business measures were not sufficient for a deeper executive summary."
        )

    rows = []
    high_terms = ("declin", "downward", "lowest", "missing", "quality", "at risk", "churn", "delivery")
    medium_terms = ("contribut", "concentration", "dependence", "growth", "movement", "top")
    recommendation_priority = {
        heading.lower(): priority for priority, heading, _ in recommendations
    }
    for heading, detail in insights:
        lowered = heading.lower() + " " + detail.lower()
        matched_priority = next(
            (priority for rec_heading, priority in recommendation_priority.items()
             if any(term in lowered for term in rec_heading.split() if len(term) > 4)),
            None
        )
        if matched_priority:
            priority = matched_priority
        elif any(term in lowered for term in high_terms):
            priority = "High"
        elif any(term in lowered for term in medium_terms):
            priority = "Medium"
        else:
            priority = "Monitor"
        rows.append({
            "Priority": priority,
            "Insight": heading,
            "Evidence": detail,
            "Suggested next step": (
                "Investigate the underlying records and validate the cause."
                if priority == "High" else
                "Compare contributing segments and monitor the next reporting period."
                if priority == "Medium" else
                "Use as a baseline and review when new data becomes available."
            )
        })
    priority_order = {"High": 0, "Medium": 1, "Monitor": 2}
    insight_table = pd.DataFrame(rows)
    if not insight_table.empty:
        insight_table["_sort"] = insight_table["Priority"].map(priority_order)
        insight_table = insight_table.sort_values(
            ["_sort", "Insight"], kind="stable"
        ).drop(columns="_sort").reset_index(drop=True)
    return " ".join(summary_parts), insight_table


def _build_business_action_plan(recommendations, roles):
    """Translate evidence-based recommendation rules into a practical action register."""
    plans = []
    for priority, heading, evidence in recommendations:
        title = heading.lower()
        if "concentration" in title or "dependence" in title:
            action, owner, timeframe, measure = (
                "Compare leading groups, identify growth barriers in smaller groups, and test a targeted campaign.",
                "Sales / Marketing", "Next 2–4 weeks", "Revenue share by category / product / market")
        elif "downward" in title:
            action, owner, timeframe, measure = (
                "Break the decline down by product, market and time; validate pricing, stock and campaign changes.",
                "Business / Sales", "Within 1 week", "Weekly sales trend and segment contribution")
        elif "growth" in title:
            action, owner, timeframe, measure = (
                "Identify growth-driving segments and check inventory, staffing and fulfilment capacity.",
                "Operations / Sales", "Next 2 weeks", "Sales growth and fulfilment performance")
        elif "experience" in title:
            action, owner, timeframe, measure = (
                "Review low-rated products or service areas and classify recurring complaint themes.",
                "Customer Experience", "Next 1–2 weeks", "Average rating and low-rating share")
        elif "delivery" in title:
            action, owner, timeframe, measure = (
                "Compare delivery lead times by region, carrier or shipping mode and investigate slow segments.",
                "Operations", "Next 2 weeks", "Average delivery time and late-delivery share")
        elif "churn" in title or "attrition" in title:
            action, owner, timeframe, measure = (
                "Compare churn outcomes across cohorts and validate possible retention interventions.",
                "Customer Success / HR", "Next 2–4 weeks", "Churn / attrition rate by cohort")
        elif "completeness" in title:
            action, owner, timeframe, measure = (
                "Prioritise missing fields used in core KPIs and trace the source of incomplete records.",
                "Data / Analytics", "Next 1–2 weeks", "Missing-value rate in critical fields")
        else:
            action, owner, timeframe, measure = (
                "Review detected fields and establish a baseline before setting a target.",
                "Analytics", "Next 2 weeks", "Coverage and quality of key business metrics")
        plans.append({
            "Priority": priority, "Finding": heading, "Recommended action": action,
            "Suggested owner": owner, "Suggested timeframe": timeframe,
            "Metric to monitor": measure, "Evidence": evidence,
        })
    return pd.DataFrame(plans)


@st.cache_data(show_spinner=False)
def _calculate_revenue_scenario(base_revenue, sales_change_pct, scenario_label):
    """Calculate a transparent what-if sales scenario from a user-selected percentage."""
    base = float(base_revenue)
    change = float(sales_change_pct)
    projected = base * (1 + change / 100)
    return {
        "scenario": scenario_label, "baseline": base, "change_pct": change,
        "projected": projected, "difference": projected - base,
    }

def _validated_role_columns(frame):
    """Return high-confidence semantic matches that passed basic type validation."""
    mapping = _semantic_column_mapping(frame)
    result = {}
    for row in mapping:
        if row["Confidence"] == "High":
            result[row["Business role"]] = row["Suggested column"]
    return result


def _render_semantic_kpis(frame, role_columns, key_prefix="semantic"):
    """Render only KPIs supported by validated fields; avoid fabricated metrics."""
    cards = []
    revenue_col = role_columns.get("Revenue / Sales")
    quantity_col = role_columns.get("Quantity / Units")
    order_col = role_columns.get("Order / Transaction ID")
    customer_col = role_columns.get("Customer")
    product_col = role_columns.get("Product")
    rating_col = role_columns.get("Rating / Review")

    if revenue_col:
        values = pd.to_numeric(frame[revenue_col], errors="coerce").dropna()
        if not values.empty:
            cards.append(("Total revenue / sales", f"{values.sum():,.2f}"))
            cards.append(("Average sales per record", f"{values.mean():,.2f}"))
    if order_col:
        cards.append(("Distinct orders / transactions", f"{frame[order_col].nunique(dropna=True):,}"))
    if customer_col:
        cards.append(("Distinct customers", f"{frame[customer_col].nunique(dropna=True):,}"))
    if product_col:
        cards.append(("Distinct products", f"{frame[product_col].nunique(dropna=True):,}"))
    if quantity_col:
        values = pd.to_numeric(frame[quantity_col], errors="coerce").dropna()
        if not values.empty:
            cards.append(("Total units", f"{values.sum():,.2f}"))
    if rating_col:
        values = pd.to_numeric(frame[rating_col], errors="coerce").dropna()
        if not values.empty:
            cards.append(("Average rating", f"{values.mean():,.2f}"))

    if cards:
        st.subheader("Business KPIs from validated column matches")
        st.caption(
            "These KPIs use high-confidence column-name matches only. "
            "Average sales per record is a row-level average, not average order value. "
            "Total sales is a row sum and should only be treated as revenue when the source field "
            "contains additive line-level sales amounts."
        )
        for offset in range(0, len(cards), 3):
            columns = st.columns(3)
            for col, (label, value) in zip(columns, cards[offset:offset + 3]):
                col.metric(label, value)
    else:
        st.info("No high-confidence business fields were found for automatic KPI calculation. Use the manual data explorer below.")


def _truthy_mask(series):
    """Interpret common binary/text outcomes without assuming a particular encoding."""
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "yes", "y", "true", "left", "exited", "churned", "late", "delayed"})


@st.cache_data(show_spinner=False)
def _domain_summary(frame):
    cols = {_clean_name(c): c for c in frame.columns}
    has = lambda *names: any(_clean_name(n) in cols for n in names)
    if has("Attrition", "EmployeeAttrition", "LeftCompany") and (
        has("Department", "JobRole", "EmployeeDepartment")
    ):
        return "HR / Employee Analytics"
    if has("Exited", "Churn", "Churned", "IsChurn") and (
        has("CreditScore", "Balance", "Geography", "AccountBalance")
    ):
        return "Banking / Customer Churn"
    if has("Late_delivery_risk", "LateDeliveryRisk", "Delivery Status",
           "Days for shipping (real)", "ShippingDays", "ActualShippingDays"):
        return "Supply Chain / Logistics"
    if has("InvoiceNo", "Invoice", "OrderID", "TransactionID") and (
        has("StockCode", "ProductID", "SKU", "ItemCode")
    ) and has("Quantity", "Qty", "UnitsSold") and has("UnitPrice", "Price", "SellingPrice"):
        return "Retail Transactions"
    return "General Dataset Analytics"


@st.cache_data(show_spinner=False)
def _detect_dataset_features(frame):
    """Identify which business analyses are supported by the columns actually uploaded."""
    aliases = {
        "Sales / revenue": ["Sales", "Revenue", "Amount", "OrderSales", "OrderItemTotal", "TotalAmount"],
        "Date trends": ["InvoiceDate", "OrderDate", "TransactionDate", "PurchaseDate", "Date", "Timestamp"],
        "Product analysis": ["Product", "ProductName", "ProductID", "SKU", "StockCode", "ItemCode"],
        "Customer analysis": ["CustomerID", "CustomerId", "CustomerName", "ClientID", "CustomerUniqueID"],
        "Geographic analysis": ["Country", "Region", "State", "City", "Market", "Geography", "PostalCode"],
        "Shipping analysis": ["ShippingDays", "ActualShippingDays", "DeliveryStatus", "LateDeliveryRisk", "ShipDate"],
        "Reviews / satisfaction": ["ReviewScore", "Rating", "Satisfaction", "CustomerRating", "FeedbackScore"],
        "Churn / retention": ["Churn", "Churned", "Exited", "Attrition", "EmployeeAttrition", "LeftCompany"],
        "Employee analysis": ["EmployeeID", "EmployeeNumber", "Department", "JobRole", "MonthlyIncome"],
    }
    cols = {_clean_name(c) for c in frame.columns}
    detected = []
    for feature, names in aliases.items():
        found = any(_clean_name(name) in cols for name in names)
        if feature == "Date trends":
            found = found or any(
                any(token in _clean_name(c) for token in ("date", "timestamp", "datetime"))
                for c in frame.columns
            )
        if feature == "Sales / revenue":
            found = found or any(
                _clean_name(c) in {"unitprice", "priceperunit"} for c in frame.columns
            )
        detected.append((feature, found))
    return detected


def _render_domain_kpis(frame, domain):
    """Render only KPIs supported by recognizable columns in the uploaded data."""
    cards = []
    if domain == "HR / Employee Analytics":
        target = _find_column(frame, ["Attrition"])
        if target:
            left = _truthy_mask(frame[target])
            cards.extend([
                ("Employees", f"{len(frame):,}"),
                ("Employees left", f"{int(left.sum()):,}"),
                ("Attrition rate", f"{left.mean() * 100:.1f}%" if len(left) else "—")
            ])
        age = _find_column(frame, ["Age"])
        if age:
            vals = pd.to_numeric(frame[age], errors="coerce").dropna()
            if len(vals): cards.append(("Average age", f"{vals.mean():.1f}"))
    elif domain == "Banking / Customer Churn":
        target = _find_column(frame, ["Exited", "Churn", "Churned"])
        active = _find_column(frame, ["IsActiveMember", "ActiveMember", "IsActive"])
        balance = _find_column(frame, ["Balance", "AccountBalance"])
        cards.append(("Customers", f"{len(frame):,}"))
        if target:
            churn = _truthy_mask(frame[target])
            cards.extend([("Customers churned", f"{int(churn.sum()):,}"),
                          ("Churn rate", f"{churn.mean() * 100:.1f}%" if len(churn) else "—")])
        if active:
            active_mask = _truthy_mask(frame[active])
            cards.append(("Active customers", f"{int(active_mask.sum()):,}"))
        if balance:
            vals = pd.to_numeric(frame[balance], errors="coerce").dropna()
            if len(vals): cards.append(("Average balance", f"{vals.mean():,.2f}"))
    elif domain == "Supply Chain / Logistics":
        shipping = _find_column(frame, ["Days for shipping (real)", "ActualShippingDays", "ShippingDays"])
        late = _find_column(frame, ["Late_delivery_risk", "LateDeliveryRisk"])
        sales = _find_column(frame, ["Sales", "Revenue", "OrderSales"])
        cards.append(("Order / line records", f"{len(frame):,}"))
        if shipping:
            vals = pd.to_numeric(frame[shipping], errors="coerce").dropna()
            if len(vals): cards.append(("Average shipping days", f"{vals.mean():.2f}"))
        if late:
            risk = _truthy_mask(frame[late])
            cards.append(("Late-delivery risk", f"{risk.mean() * 100:.1f}%" if len(risk) else "—"))
        if sales:
            vals = pd.to_numeric(frame[sales], errors="coerce").dropna()
            if len(vals): cards.append(("Sales total", f"{vals.sum():,.2f}"))
    else:
        cards.append(("Records", f"{len(frame):,}"))
        for label, aliases, agg in [
            ("Total sales", ["Sales", "Revenue", "Amount"], "sum"),
            ("Average value", ["Amount", "Price", "Balance"], "mean")
        ]:
            col = _find_column(frame, aliases)
            if col:
                vals = pd.to_numeric(frame[col], errors="coerce").dropna()
                if len(vals): cards.append((label, f"{getattr(vals, agg)():,.2f}"))
    if cards:
        cols = st.columns(min(4, len(cards)))
        for i, (label, value) in enumerate(cards):
            cols[i % len(cols)].metric(label, value)


def _render_adaptive_workspace(tables):
    """Dataset-aware analytics for non-legacy and multi-table uploads."""
    st.title("RetailIQ AI")
    st.subheader("Adaptive Analytics & Data Explorer")
    st.caption(
        "RetailIQ profiles the uploaded data, detects likely business context and only displays analyses supported by available fields."
    )
    st.success("Adaptive mode enabled — the original Online Retail dashboard logic has not been applied.")

    table_names = list(tables.keys())
    if not table_names:
        st.error("No readable tables were found.")
        return

    inventory = pd.DataFrame([
        {"Table": name, "Rows": len(tables[name]), "Columns": len(tables[name].columns)}
        for name in table_names
    ])
    with st.expander("Uploaded data inventory", expanded=len(table_names) > 1):
        st.dataframe(inventory, use_container_width=True, hide_index=True)

    working, base_name = None, None
    if len(table_names) == 1:
        base_name = table_names[0]
        working = tables[base_name].copy()
    else:
        st.subheader("Choose a primary table")
        default_base = max(table_names, key=lambda n: len(tables[n]))
        base_name = st.selectbox("Primary table for analysis", table_names,
                                 index=table_names.index(default_base), key="adaptive_primary_table")
        working = tables[base_name].copy()
        st.subheader("Optional table relationships")
        st.caption("Join matching keys only when they represent the same entity. Many-to-many joins can multiply records.")
        for idx, other_name in enumerate([n for n in table_names if n != base_name]):
            other = tables[other_name]
            with st.expander(f"Join {other_name}", expanded=False):
                common = [c for c in working.columns if c in other.columns]
                include = st.checkbox(f"Include {other_name}", value=False, key=f"adaptive_join_include_{idx}")
                if not include:
                    st.caption(f"Potential common columns: {', '.join(common[:8]) if common else 'None detected'}")
                    continue
                left_options, right_options = list(working.columns), list(other.columns)
                suggested_left = common[0] if common else left_options[0]
                suggested_right = suggested_left if suggested_left in right_options else right_options[0]
                c1, c2 = st.columns(2)
                left_key = c1.selectbox("Primary-table key", left_options, index=left_options.index(suggested_left), key=f"adaptive_left_key_{idx}")
                right_key = c2.selectbox("Related-table key", right_options, index=right_options.index(suggested_right), key=f"adaptive_right_key_{idx}")
                join_how = st.selectbox("Join type", ["left", "inner", "outer"], key=f"adaptive_join_how_{idx}")
                if working[left_key].duplicated().any() and other[right_key].duplicated().any():
                    st.warning("Both selected keys contain duplicates. This many-to-many join may multiply rows.")
                if st.button(f"Apply join: {other_name}", key=f"adaptive_apply_join_{idx}"):
                    try:
                        joined = working.merge(other, left_on=left_key, right_on=right_key, how=join_how,
                                               suffixes=("", f"_{_clean_name(other_name)[:12]}"))
                        st.session_state["adaptive_joined_frame"] = joined
                        st.session_state["adaptive_joined_base"] = base_name
                        st.success(f"Join applied: {len(joined):,} rows and {len(joined.columns):,} columns.")
                    except Exception as exc:
                        st.error(f"Join could not be completed: {exc}")
                    st.rerun()
        if st.session_state.get("adaptive_joined_base") == base_name and "adaptive_joined_frame" in st.session_state:
            working = st.session_state["adaptive_joined_frame"].copy()

    if working is None or working.empty:
        st.warning("The selected table is empty.")
        return

    # Batch 5: interactive filters are applied only to the selected working copy.
    # The uploaded source tables remain unchanged.
    unfiltered_rows = len(working)
    filter_profile = _profile_table(working)
    st.subheader("Interactive Data Filters")
    st.caption(
        "Filter the selected table before running the analytics below. "
        "Filters affect this analysis view and its exports, not the uploaded source files."
    )
    filtered = working.copy()
    filter_cols = st.columns(2)

    available_date_cols = filter_profile["dates"]
    if available_date_cols:
        with filter_cols[0]:
            filter_date_col = st.selectbox(
                "Date field", ["No date filter"] + available_date_cols,
                key=f"adaptive_filter_date_col_{_clean_name(base_name)}"
            )
        if filter_date_col != "No date filter":
            parsed_dates = pd.to_datetime(filtered[filter_date_col], errors="coerce")
            valid_dates = parsed_dates.dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                with filter_cols[1]:
                    selected_dates = st.date_input(
                        "Date range", value=(min_date, max_date),
                        min_value=min_date, max_value=max_date,
                        key=f"adaptive_filter_date_range_{_clean_name(base_name)}"
                    )
                if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
                    start_date, end_date = selected_dates
                    filtered = filtered.loc[
                        parsed_dates.dt.date.between(start_date, end_date)
                    ].copy()
                elif isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 1:
                    filtered = filtered.loc[parsed_dates.dt.date == selected_dates[0]].copy()
            else:
                st.info("The selected date column has no valid dates to filter.")
    else:
        st.info("No date column was detected; date filtering is unavailable.")

    filterable_categories = [
        col for col in filter_profile["categorical"]
        if 2 <= int(filter_profile["unique"].get(col, 0)) <= 50
    ]
    if filterable_categories:
        category_col = st.selectbox(
            "Category field", ["No category filter"] + filterable_categories,
            key=f"adaptive_filter_category_col_{_clean_name(base_name)}"
        )
        if category_col != "No category filter":
            category_values = sorted(
                filtered[category_col].dropna().astype(str).unique().tolist()
            )
            selected_categories = st.multiselect(
                "Keep selected categories (leave blank to include all)",
                category_values,
                key=f"adaptive_filter_category_values_{_clean_name(base_name)}"
            )
            if selected_categories:
                filtered = filtered.loc[
                    filtered[category_col].astype(str).isin(selected_categories)
                ].copy()

    # Batch 6: optional numeric range filter.
    filter_numeric_cols = [
        col for col in filter_profile["numeric"]
        if pd.to_numeric(filtered[col], errors="coerce").notna().sum() > 1
    ]
    if filter_numeric_cols:
        st.markdown("**Optional numeric range filter**")
        numeric_filter_col = st.selectbox(
            "Numeric field", ["No numeric filter"] + filter_numeric_cols,
            key=f"adaptive_filter_numeric_col_{_clean_name(base_name)}"
        )
        if numeric_filter_col != "No numeric filter":
            numeric_values = pd.to_numeric(filtered[numeric_filter_col], errors="coerce").dropna()
            numeric_min, numeric_max = float(numeric_values.min()), float(numeric_values.max())
            if numeric_min < numeric_max:
                numeric_step = (numeric_max - numeric_min) / 100
                selected_range = st.slider(
                    f"Keep {numeric_filter_col} within",
                    min_value=numeric_min,
                    max_value=numeric_max,
                    value=(numeric_min, numeric_max),
                    step=numeric_step if numeric_step > 0 else 0.01,
                    key=f"adaptive_filter_numeric_range_{_clean_name(base_name)}"
                )
                numeric_series = pd.to_numeric(filtered[numeric_filter_col], errors="coerce")
                filtered = filtered.loc[
                    numeric_series.between(selected_range[0], selected_range[1])
                ].copy()
            else:
                st.caption("The selected numeric field has no range to narrow.")
    working = filtered
    if working.empty:
        st.warning("No records match the selected filters. Adjust the filters to continue.")
        return
    st.caption(
        f"Analysis rows: {len(working):,} of {unfiltered_rows:,} "
        f"({len(working) / max(1, unfiltered_rows) * 100:.1f}% retained)."
    )

    # Batch 7: surface quality diagnostics and provide an optional, non-destructive
    # duplicate-removal view for downstream analysis.
    quality_profile = _profile_table(working)
    quality_report = _build_data_quality_report(working, quality_profile)
    total_cells = max(1, working.size)
    missing_cells = int(working.isna().sum().sum())
    duplicate_rows = int(working.duplicated().sum())
    st.subheader("Data Quality Monitor")
    quality_metrics = st.columns(4)
    quality_metrics[0].metric("Rows", f"{len(working):,}")
    quality_metrics[1].metric("Columns", f"{len(working.columns):,}")
    quality_metrics[2].metric("Missing cells", f"{missing_cells:,}",
                              f"{missing_cells / total_cells * 100:.1f}% of all cells")
    quality_metrics[3].metric("Duplicate rows", f"{duplicate_rows:,}")
    with st.expander("Column-level quality diagnostics", expanded=False):
        st.dataframe(quality_report, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Data Quality Report (CSV)",
            data=_cached_csv_export(quality_report),
            file_name=f"{_clean_name(str(base_name)) or 'retailiq'}_data_quality_report.csv",
            mime="text/csv",
            key=f"adaptive_quality_report_download_{_clean_name(base_name)}"
        )

    if duplicate_rows:
        remove_duplicates_for_analysis = st.checkbox(
            f"Exclude {duplicate_rows:,} exact duplicate rows from this analysis view",
            value=False,
            key=f"adaptive_remove_duplicates_{_clean_name(base_name)}"
        )
        if remove_duplicates_for_analysis:
            before_dedup = len(working)
            working = working.drop_duplicates().copy()
            st.info(
                f"Analysis view now contains {len(working):,} rows "
                f"({before_dedup - len(working):,} duplicates excluded). "
                "The uploaded source data has not been changed."
            )

    profile = _profile_table(working)
    date_options, numeric_options, category_options = profile["dates"], profile["numeric"], profile["categorical"]

    # Batch 8: exception monitoring using transparent statistical rules.
    st.subheader("Anomaly & Exception Monitor")
    st.caption(
        "Identify unusually high or low numeric values in the current analysis view. "
        "These are statistical flags for review, not proof of errors or fraud."
    )
    anomaly_candidates = [
        col for col in numeric_options
        if pd.to_numeric(working[col], errors="coerce").notna().sum() >= 8
    ]
    if anomaly_candidates:
        anomaly_controls = st.columns(2)
        with anomaly_controls[0]:
            anomaly_column = st.selectbox(
                "Metric to inspect", anomaly_candidates,
                key=f"adaptive_anomaly_column_{_clean_name(base_name)}"
            )
        with anomaly_controls[1]:
            anomaly_method = st.selectbox(
                "Detection method", ["Interquartile range (IQR)", "Z-score"],
                key=f"adaptive_anomaly_method_{_clean_name(base_name)}"
            )
        anomaly_values = pd.to_numeric(working[anomaly_column], errors="coerce")
        valid_anomaly_values = anomaly_values.dropna()
        if anomaly_method == "Interquartile range (IQR)":
            q1, q3 = valid_anomaly_values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            anomaly_mask = anomaly_values.lt(lower_bound) | anomaly_values.gt(upper_bound)
            method_note = "Values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR are flagged."
        else:
            mean_value = valid_anomaly_values.mean()
            std_value = valid_anomaly_values.std(ddof=0)
            if std_value > 0:
                anomaly_score = (anomaly_values - mean_value) / std_value
                anomaly_mask = anomaly_score.abs() > 3
            else:
                anomaly_mask = pd.Series(False, index=working.index)
            lower_bound = mean_value - 3 * std_value
            upper_bound = mean_value + 3 * std_value
            method_note = "Values more than 3 standard deviations from the mean are flagged."
        anomaly_mask = anomaly_mask.fillna(False)
        flagged_count = int(anomaly_mask.sum())
        anomaly_metrics = st.columns(3)
        anomaly_metrics[0].metric("Flagged records", f"{flagged_count:,}")
        anomaly_metrics[1].metric(
            "Share flagged", f"{flagged_count / max(1, len(working)) * 100:.2f}%"
        )
        anomaly_metrics[2].metric(
            "Records reviewed", f"{int(anomaly_values.notna().sum()):,}"
        )
        st.caption(
            f"{method_note} Reference bounds: {lower_bound:,.4g} to {upper_bound:,.4g}."
        )
        if flagged_count:
            anomaly_rows = working.loc[anomaly_mask].copy()
            anomaly_rows.insert(0, "Source row index", anomaly_rows.index.astype(str))
            st.dataframe(anomaly_rows.head(500), use_container_width=True, hide_index=True)
            if flagged_count > 500:
                st.caption("Showing the first 500 flagged records; the CSV export includes all flagged records.")
            st.download_button(
                "Download Flagged Records (CSV)",
                data=_cached_csv_export(anomaly_rows),
                file_name=f"{_clean_name(str(base_name)) or 'retailiq'}_anomaly_flags.csv",
                mime="text/csv",
                key=f"adaptive_anomaly_download_{_clean_name(base_name)}"
            )
        else:
            st.success("No records were flagged by the selected method.")
    else:
        st.info("Anomaly monitoring needs a numeric field with at least eight valid values.")
    domain = _domain_summary(working)
    st.subheader("Data quality and structure")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Records", f"{len(working):,}")
    q2.metric("Columns", f"{len(working.columns):,}")
    q3.metric("Business measures", f"{len(numeric_options):,}")
    q4.metric("Date fields", f"{len(date_options):,}")

    profile_rows = []
    for col in working.columns:
        detected = ("Date" if col in profile["dates"] else
                    "Identifier / key" if col in profile["identifiers"] else
                    "Geographic coordinate" if col in profile["geo"] else
                    "Constant / metadata" if col in profile["metadata"] else
                    "Numeric measure" if col in profile["numeric"] else
                    "Category / flag" if col in profile["categorical"] else "Text / high-cardinality")
        profile_rows.append({"Column": str(col), "Detected type": detected,
                             "Missing": profile["missing"][col], "Unique values": profile["unique"][col]})
    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    st.subheader("Suggested business column mapping")
    st.caption(
        "RetailIQ matches common naming variations to business concepts. "
        "These are suggestions only; confirm them before using a field in calculations."
    )
    semantic_mapping_rows = _semantic_column_mapping(working)
    semantic_mapping = pd.DataFrame(semantic_mapping_rows)
    st.dataframe(semantic_mapping, use_container_width=True, hide_index=True)
    validated_roles = _validated_role_columns(working)
    _render_semantic_kpis(working, validated_roles, key_prefix=_clean_name(base_name))

    st.subheader("Custom Business Breakdown")
    st.caption(
        "Compare a numeric measure across a category detected in your dataset. "
        "The chart and table use the currently filtered analysis data."
    )
    breakdown_categories = [
        col for col in profile["categorical"]
        if 2 <= int(profile["unique"].get(col, 0)) <= 100
    ]
    breakdown_measures = list(profile["numeric"])
    if breakdown_categories and breakdown_measures:
        breakdown_controls = st.columns(3)
        with breakdown_controls[0]:
            breakdown_category = st.selectbox(
                "Group by", breakdown_categories,
                key=f"adaptive_breakdown_category_{_clean_name(base_name)}"
            )
        with breakdown_controls[1]:
            breakdown_measure = st.selectbox(
                "Measure", breakdown_measures,
                key=f"adaptive_breakdown_measure_{_clean_name(base_name)}"
            )
        with breakdown_controls[2]:
            breakdown_aggregation = st.selectbox(
                "Aggregation", ["Sum", "Average", "Median", "Count"],
                key=f"adaptive_breakdown_aggregation_{_clean_name(base_name)}"
            )
        breakdown_source = working[[breakdown_category, breakdown_measure]].copy()
        breakdown_source[breakdown_measure] = pd.to_numeric(
            breakdown_source[breakdown_measure], errors="coerce"
        )
        if breakdown_aggregation == "Sum":
            breakdown_table = breakdown_source.groupby(
                breakdown_category, dropna=False
            )[breakdown_measure].sum().reset_index(name="Value")
        elif breakdown_aggregation == "Average":
            breakdown_table = breakdown_source.groupby(
                breakdown_category, dropna=False
            )[breakdown_measure].mean().reset_index(name="Value")
        elif breakdown_aggregation == "Median":
            breakdown_table = breakdown_source.groupby(
                breakdown_category, dropna=False
            )[breakdown_measure].median().reset_index(name="Value")
        else:
            breakdown_table = breakdown_source.groupby(
                breakdown_category, dropna=False
            )[breakdown_measure].count().reset_index(name="Value")
        breakdown_table[breakdown_category] = breakdown_table[breakdown_category].astype(str)
        breakdown_table = breakdown_table.sort_values("Value", ascending=False).head(20)
        if not breakdown_table.empty:
            st.plotly_chart(
                px.bar(
                    breakdown_table, x=breakdown_category, y="Value",
                    title=f"{breakdown_aggregation} of {breakdown_measure} by {breakdown_category}",
                    template="plotly_white"
                ),
                use_container_width=True,
                key=f"adaptive_custom_breakdown_{_clean_name(base_name)}"
            )
            st.dataframe(breakdown_table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Custom Breakdown (CSV)",
                data=_cached_csv_export(breakdown_table),
                file_name=f"{_clean_name(str(base_name)) or 'retailiq'}_custom_breakdown.csv",
                mime="text/csv",
                key=f"adaptive_breakdown_download_{_clean_name(base_name)}"
            )
        else:
            st.info("No valid grouped values are available for this selection.")
    else:
        st.info("Custom breakdown needs at least one detected category and one numeric measure.")

    business_insights = _generate_business_insights(working, validated_roles)
    business_recommendations = _generate_business_recommendations(working, validated_roles)
    executive_summary, prioritized_insights = _build_executive_insight_summary(
        working, validated_roles, business_insights, business_recommendations
    )
    st.subheader("Executive Summary & Prioritized Insights")
    st.caption(
        "A concise, evidence-linked overview generated from validated dataset fields and transparent rules. "
        "Priority labels are review guidance, not a substitute for business judgement."
    )
    with st.container(border=True):
        st.markdown("**Executive overview**")
        st.write(executive_summary)
    if not prioritized_insights.empty:
        st.dataframe(prioritized_insights, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Prioritized Insights (CSV)",
            data=_cached_csv_export(prioritized_insights),
            file_name=f"{_clean_name(str(base_name)) or 'retailiq'}_prioritized_insights.csv",
            mime="text/csv",
            key=f"prioritized_insights_download_{_clean_name(base_name)}"
        )
    else:
        st.info("No automated insight rules were triggered for the selected data.")

    st.subheader("Automated Business Insights")
    st.caption(
        "Observations are calculated from detected fields in the selected table. "
        "They describe the uploaded data and do not establish causation."
    )
    if business_insights:
        for idx in range(0, len(business_insights), 2):
            insight_cols = st.columns(2)
            for col, (heading, detail) in zip(insight_cols, business_insights[idx:idx + 2]):
                with col:
                    st.markdown(f"**{heading}**")
                    st.write(detail)
    else:
        st.info(
            "There are not enough high-confidence business fields to generate automated insights. "
            "Review the suggested column mapping or use the manual explorer."
        )

    st.subheader("Business Advisor — Recommended Actions")
    st.caption(
        "Recommendations use transparent rules and validated columns in this dataset. "
        "They are prompts for investigation, not causal conclusions or guaranteed predictions."
    )
    for priority, heading, detail in business_recommendations:
        with st.container(border=True):
            p1, p2 = st.columns([1, 5])
            with p1:
                if priority == "High":
                    st.error("High priority")
                elif priority == "Medium":
                    st.warning("Medium priority")
                else:
                    st.info("Guidance")
            with p2:
                st.markdown(f"**{heading}**")
                st.write(detail)

    st.subheader("Customer Intelligence — RFM & Cohort Retention")
    st.caption(
        "This section appears only when high-confidence customer and date fields are available. "
        "RFM scores are relative rankings within the selected dataset; they are not churn predictions."
    )
    rfm_table, rfm_note = _calculate_rfm_analysis(working, validated_roles)
    cohort_table, cohort_note = _calculate_customer_cohorts(working, validated_roles)

    if not rfm_table.empty:
        rfm_customer_col = validated_roles["Customer"]
        rfm_kpis = st.columns(4)
        rfm_kpis[0].metric("Customers analysed", f"{len(rfm_table):,}")
        rfm_kpis[1].metric("Champions", f"{int((rfm_table['Segment'] == 'Champions').sum()):,}")
        rfm_kpis[2].metric("At risk", f"{int(rfm_table['Segment'].isin(['At risk', 'Hibernating']).sum()):,}")
        rfm_kpis[3].metric("Average monetary value", f"{rfm_table['Monetary'].mean():,.2f}")
        st.caption(
            f"Snapshot date: {(pd.to_datetime(working[validated_roles['Date / Timestamp']], errors='coerce').max()).date()}. "
            + rfm_note
        )
        segment_summary = (
            rfm_table.groupby("Segment", as_index=False)
            .agg(Customers=(rfm_customer_col, "nunique"), Monetary=("Monetary", "sum"))
            .sort_values("Customers", ascending=False)
        )
        rfm_cols = st.columns(2)
        with rfm_cols[0]:
            st.plotly_chart(
                px.bar(segment_summary, x="Segment", y="Customers", title="Customers by RFM segment",
                       template="plotly_white"),
                use_container_width=True, key=f"rfm_segments_{_clean_name(base_name)}"
            )
        with rfm_cols[1]:
            st.plotly_chart(
                px.bar(segment_summary, x="Segment", y="Monetary", title="Monetary value by RFM segment",
                       template="plotly_white"),
                use_container_width=True, key=f"rfm_value_{_clean_name(base_name)}"
            )
        st.dataframe(
            rfm_table.sort_values(["RFM_score", "Monetary"], ascending=False).head(100),
            use_container_width=True, hide_index=True
        )
        rfm_csv = _cached_csv_export(rfm_table)
        st.download_button(
            "Download RFM Customer Segments (CSV)", data=rfm_csv,
            file_name=f"{_clean_name(str(base_name)) or 'retailiq'}_rfm_segments.csv",
            mime="text/csv", key=f"rfm_download_{_clean_name(base_name)}"
        )
    else:
        st.info(rfm_note)

    if not cohort_table.empty:
        st.markdown("**Monthly customer cohort retention**")
        st.caption(cohort_note)
        st.dataframe(
            cohort_table.style.format("{:.1f}%").background_gradient(cmap="Blues", axis=None),
            use_container_width=True
        )
        cohort_csv = cohort_table.to_csv()
        st.download_button(
            "Download Cohort Retention (CSV)", data=cohort_csv,
            file_name=f"{_clean_name(str(base_name)) or 'retailiq'}_cohort_retention.csv",
            mime="text/csv", key=f"cohort_download_{_clean_name(base_name)}"
        )
    else:
        st.caption(cohort_note)

    st.subheader("Advanced Customer Intelligence")
    st.caption(
        "Understand repeat purchasing, historical customer value and revenue concentration. "
        "These are descriptive metrics, not predictions of future customer lifetime value."
    )
    customer_intel, customer_intel_note = _calculate_advanced_customer_intelligence(
        working, validated_roles
    )
    if customer_intel:
        customer_kpis = st.columns(4)
        customer_kpis[0].metric("Repeat customer rate", f"{customer_intel['repeat_rate']:.1f}%")
        customer_kpis[1].metric("Historical revenue / customer",
                                f"{customer_intel['revenue_per_customer']:,.2f}")
        customer_kpis[2].metric("Average order value",
                                f"{customer_intel['average_order_value']:,.2f}")
        customer_kpis[3].metric(
            f"Top 20% customer revenue share",
            f"{customer_intel['top_customer_revenue_share']:.1f}%"
        )
        st.caption(customer_intel["frequency_note"])
        customer_type_counts = (
            customer_intel["customers"].groupby("Customer type", as_index=False)
            .agg(Customers=(validated_roles["Customer"], "nunique"))
        )
        customer_cols = st.columns(2)
        with customer_cols[0]:
            st.plotly_chart(
                px.pie(
                    customer_type_counts, names="Customer type", values="Customers",
                    title="One-time vs repeat customers", hole=0.45,
                    template="plotly_white"
                ),
                use_container_width=True,
                key=f"advanced_customer_repeat_{_clean_name(base_name)}"
            )
        with customer_cols[1]:
            pareto_view = customer_intel["customers"].head(20).copy()
            pareto_view[validated_roles["Customer"]] = pareto_view[
                validated_roles["Customer"]
            ].astype(str)
            st.plotly_chart(
                px.bar(
                    pareto_view, x=validated_roles["Customer"], y="Revenue share (%)",
                    title="Revenue contribution — top 20 customers",
                    template="plotly_white"
                ),
                use_container_width=True,
                key=f"advanced_customer_pareto_{_clean_name(base_name)}"
            )
        st.write(
            f"**{customer_intel['repeat_customer_count']:,}** of "
            f"**{customer_intel['customer_count']:,}** customers are classified as repeat customers. "
            f"The top **{customer_intel['top_customer_count']:,}** customers account for "
            f"**{customer_intel['top_customer_revenue_share']:.1f}%** of recorded revenue."
        )
        customer_table = customer_intel["customers"]
        st.dataframe(customer_table.head(100), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Customer Intelligence (CSV)",
            data=_cached_csv_export(customer_table),
            file_name=f"{_clean_name(str(base_name)) or 'retailiq'}_customer_intelligence.csv",
            mime="text/csv",
            key=f"advanced_customer_download_{_clean_name(base_name)}"
        )
    else:
        st.info(customer_intel_note)

    st.subheader("Predictive Analytics — Churn & Sales Forecast")
    st.caption(
        "Models are shown only when the uploaded dataset contains suitable validated fields. "
        "Review holdout metrics before using predictions in business decisions."
    )

    churn_result, churn_note = _train_churn_classification(working, validated_roles)
    if churn_result:
        st.markdown("**Customer / employee churn classification**")
        st.caption(churn_note)
        churn_metrics = st.columns(5)
        churn_metrics[0].metric("Accuracy", f"{churn_result['accuracy'] * 100:.1f}%")
        churn_metrics[1].metric("Precision", f"{churn_result['precision'] * 100:.1f}%")
        churn_metrics[2].metric("Recall", f"{churn_result['recall'] * 100:.1f}%")
        churn_metrics[3].metric("F1 score", f"{churn_result['f1']:.3f}")
        churn_metrics[4].metric("ROC AUC", f"{churn_result['auc']:.3f}" if np.isfinite(churn_result["auc"]) else "—")
        st.write(
            f"Evaluated {churn_result['test_records']:,} holdout records from "
            f"{churn_result['records']:,} labelled records using {len(churn_result['features']):,} features."
        )
        st.write(
            f"Confusion matrix counts — true positives: {churn_result['tp']:,}, "
            f"true negatives: {churn_result['tn']:,}, false positives: {churn_result['fp']:,}, "
            f"false negatives: {churn_result['fn']:,}."
        )
        st.caption("This model estimates patterns in historical labelled data. It does not establish causes or guarantee that an individual will churn.")
    else:
        st.info(f"Churn model unavailable: {churn_note}")

    predictive_date = validated_roles.get("Date / Timestamp")
    predictive_revenue = validated_roles.get("Revenue / Sales")
    if predictive_date and predictive_revenue:
        forecast_input = working[[predictive_date, predictive_revenue]].copy()
        forecast_input[predictive_date] = pd.to_datetime(forecast_input[predictive_date], errors="coerce")
        forecast_input[predictive_revenue] = pd.to_numeric(forecast_input[predictive_revenue], errors="coerce")
        forecast_input = forecast_input.dropna()
        if not forecast_input.empty:
            daily_sales = forecast_input.set_index(predictive_date)[predictive_revenue].resample("D").sum()
            if len(daily_sales) >= 35:
                st.markdown("**Sales forecast with holdout evaluation**")
                backtest, backtest_note = _evaluate_time_series_forecast(daily_sales)
                if backtest:
                    eval_cols = st.columns(3)
                    eval_cols[0].metric("Holdout MAE", f"{backtest['mae']:,.2f}")
                    eval_cols[1].metric("Holdout RMSE", f"{backtest['rmse']:,.2f}")
                    eval_cols[2].metric("Holdout MAPE", f"{backtest['mape']:.1f}%" if np.isfinite(backtest["mape"]) else "—")
                    comparison = pd.DataFrame({
                        "Date": backtest["actual"].index,
                        "Actual": backtest["actual"].values,
                        "Predicted": backtest["predicted"].values,
                    })
                    st.plotly_chart(
                        px.line(comparison, x="Date", y=["Actual", "Predicted"],
                                title="Forecast backtest: actual vs predicted", template="plotly_white"),
                        use_container_width=True, key=f"adaptive_forecast_backtest_{_clean_name(base_name)}"
                    )
                    st.caption(backtest_note)
                try:
                    future = _forecast_from_daily_series(daily_sales, 30)
                    future_frame = pd.DataFrame({"Date": future.index, "Forecast sales": future.values})
                    st.metric("Next 30-day forecast", f"{future.sum():,.2f}")
                    st.plotly_chart(
                        px.line(future_frame, x="Date", y="Forecast sales", markers=True,
                                title="Next 30 days — forecast", template="plotly_white"),
                        use_container_width=True, key=f"adaptive_forecast_future_{_clean_name(base_name)}"
                    )
                except Exception as exc:
                    st.warning(f"Future sales forecast could not be generated: {exc}")
            else:
                st.info("Sales forecasting needs at least 35 daily observations for backtesting.")
    else:
        st.info("Adaptive sales forecasting needs high-confidence Date and Revenue/Sales columns.")

    st.subheader("Business Scenario Planner")
    st.caption(
        "Explore a what-if change to the selected dataset's total sales. "
        "This is a planning assumption, not a model prediction."
    )
    scenario_measure = validated_roles.get("Revenue / Sales")
    if scenario_measure:
        scenario_values = pd.to_numeric(working[scenario_measure], errors="coerce").dropna()
        if len(scenario_values):
            baseline_revenue = float(scenario_values.sum())
            s1, s2 = st.columns([2, 1])
            with s1:
                scenario_change = st.slider(
                    "Assumed change in sales (%)", -50, 100, 10, 5,
                    key=f"scenario_sales_change_{_clean_name(base_name)}"
                )
            with s2:
                scenario_label = st.selectbox(
                    "Scenario", ["Conservative", "Base case", "Optimistic"], index=1,
                    key=f"scenario_label_{_clean_name(base_name)}"
                )
            scenario = _calculate_revenue_scenario(
                baseline_revenue, scenario_change, scenario_label
            )
            scenario_metrics = st.columns(3)
            scenario_metrics[0].metric("Baseline sales", f"{scenario['baseline']:,.2f}")
            scenario_metrics[1].metric(
                "Scenario sales", f"{scenario['projected']:,.2f}",
                f"{scenario['change_pct']:+.1f}%"
            )
            scenario_metrics[2].metric(
                "Estimated difference", f"{scenario['difference']:+,.2f}"
            )
            scenario_chart = pd.DataFrame({
                "Measure": ["Baseline", scenario_label],
                "Sales": [scenario["baseline"], scenario["projected"]],
            })
            st.plotly_chart(
                px.bar(scenario_chart, x="Measure", y="Sales",
                       title="Baseline vs what-if sales scenario", template="plotly_white"),
                use_container_width=True, key=f"scenario_chart_{_clean_name(base_name)}"
            )
            st.caption(
                "The scenario applies the selected percentage to historical total sales. "
                "It does not estimate demand elasticity, costs, profit or causal effects."
            )
            st.download_button(
                "Download Scenario Summary (CSV)", data=scenario_chart.to_csv(index=False),
                file_name=f"{_clean_name(str(base_name)) or 'retailiq'}_scenario_summary.csv",
                mime="text/csv", key=f"scenario_download_{_clean_name(base_name)}"
            )
        else:
            st.info("No valid numeric sales values are available for scenario planning.")
    else:
        st.info("Scenario planning needs a high-confidence Revenue / Sales column.")

    st.subheader("Business Action Plan")
    st.caption(
        "A practical action register generated from the rule-based findings above. "
        "Owners and timeframes are suggestions and should be confirmed by your team."
    )
    action_plan = _build_business_action_plan(business_recommendations, validated_roles)
    st.dataframe(
        action_plan[["Priority", "Finding", "Recommended action", "Suggested owner",
                     "Suggested timeframe", "Metric to monitor"]],
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "Download Business Action Plan (CSV)", data=_cached_csv_export(action_plan),
        file_name=f"{_clean_name(str(base_name)) or 'retailiq'}_action_plan.csv",
        mime="text/csv", key=f"action_plan_download_{_clean_name(base_name)}"
    )

    st.subheader("Export Business Report")
    st.caption(
        "Download a concise text report containing the selected dataset profile, "
        "automated insights, and recommended actions."
    )
    report_lines = [
        "RETAILIQ AI — BUSINESS ANALYSIS REPORT",
        "=" * 44,
        f"Dataset: {base_name}",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Records: {len(working):,}",
        f"Columns: {len(working.columns):,}",
        f"Duplicate rows: {profile['duplicate_rows']:,}",
        f"Missing cells: {int(working.isna().sum().sum()):,}",
        "",
        "VALIDATED BUSINESS FIELD MAPPINGS",
        "-" * 34,
    ]
    if validated_roles:
        report_lines.extend([f"{role}: {column}" for role, column in validated_roles.items()])
    else:
        report_lines.append("No high-confidence semantic mappings were available.")
    report_lines.extend([
        "",
        "COLUMN-LEVEL DATA QUALITY",
        "-" * 26,
        "Column | Detected type | Missing values | Unique values",
    ])
    for row in profile_rows:
        report_lines.append(
            f"{row['Column']} | {row['Detected type']} | "
            f"{row['Missing']:,} | {row['Unique values']:,}"
        )
    report_lines.extend(["", "AUTOMATED BUSINESS INSIGHTS", "-" * 28])
    if business_insights:
        for heading, detail in business_insights:
            report_lines.extend([f"{heading}: {detail}", ""])
    else:
        report_lines.append("Not enough validated fields to generate insights.")
    report_lines.extend(["", "BUSINESS ADVISOR — RECOMMENDED ACTIONS", "-" * 39])
    if business_recommendations:
        for priority, heading, detail in business_recommendations:
            report_lines.extend([f"[{priority}] {heading}", detail, ""])
    else:
        report_lines.append("No rule-based recommendations were triggered for this dataset.")
    report_lines.extend(["", "SUGGESTED BUSINESS ACTION PLAN", "-" * 30])
    for _, action_row in action_plan.iterrows():
        report_lines.append(
            f"[{action_row['Priority']}] {action_row['Finding']} | "
            f"Owner: {action_row['Suggested owner']} | "
            f"Timeframe: {action_row['Suggested timeframe']} | "
            f"Action: {action_row['Recommended action']} | "
            f"Monitor: {action_row['Metric to monitor']}"
        )
    report_lines.extend([
        "",
        "NOTE",
        "This report summarizes patterns in the selected data. Recommendations are rule-based prompts for investigation; they do not establish causation or guarantee future results. Validate field mappings and business context before making decisions.",
    ])
    report_text = "\n".join(report_lines)
    report_filename = f"{_clean_name(str(base_name)) or 'retailiq'}_business_report.txt"
    st.download_button(
        "Download Business Report",
        data=report_text,
        file_name=report_filename,
        mime="text/plain",
        key=f"business_report_{_clean_name(str(base_name))}",
        use_container_width=True,
    )

    st.subheader("Export Data Dictionary")
    st.caption(
        "Download a CSV reference for the selected dataset, including detected column types, "
        "missing-value rates, unique-value counts and suggested business roles."
    )
    role_lookup = {}
    for mapping_row in semantic_mapping_rows:
        suggested = mapping_row["Suggested column"]
        if suggested != "Not detected":
            role_lookup.setdefault(suggested, []).append(
                f"{mapping_row['Business role']} ({mapping_row['Confidence']})"
            )
    dictionary_rows = []
    for row in profile_rows:
        missing_count = int(row["Missing"])
        dictionary_rows.append({
            "Column": row["Column"],
            "Detected type": row["Detected type"],
            "Missing values": missing_count,
            "Missing %": round(missing_count / max(1, len(working)) * 100, 2),
            "Unique values": int(row["Unique values"]),
            "Suggested business role": "; ".join(role_lookup.get(row["Column"], [])) or "Not detected",
        })
    dictionary_csv = _cached_csv_export(pd.DataFrame(dictionary_rows))
    dictionary_filename = f"{_clean_name(str(base_name)) or 'retailiq'}_data_dictionary.csv"
    st.download_button(
        "Download Data Dictionary (CSV)",
        data=dictionary_csv,
        file_name=dictionary_filename,
        mime="text/csv",
        key=f"data_dictionary_{_clean_name(str(base_name))}",
        use_container_width=True,
    )

    st.subheader("Export Selected Analysis Dataset")
    st.caption(
        "Download the exact table currently used for Adaptive Analytics. "
        "If you applied table joins, the export includes the joined result. "
        "Source values are not automatically cleaned or changed."
    )
    export_csv = _cached_csv_export(working)
    export_filename = f"{_clean_name(str(base_name)) or 'retailiq'}_analysis_dataset.csv"
    st.download_button(
        "Download Analysis Dataset (CSV)",
        data=export_csv,
        file_name=export_filename,
        mime="text/csv",
        key=f"analysis_dataset_{_clean_name(str(base_name))}",
        use_container_width=True,
    )

    st.subheader("Automatically detected analysis features")
    st.caption("Feature availability is inferred from column names. Review the detected fields before using the results for business decisions.")
    detected_features = _detect_dataset_features(working)
    available_features = [name for name, available in detected_features if available]
    feature_cols = st.columns(3)
    for idx, (feature_name, is_available) in enumerate(detected_features):
        target_col = feature_cols[idx % 3]
        if is_available:
            target_col.success(f"Available · {feature_name}")
        else:
            target_col.info(f"Not detected · {feature_name}")
    if not available_features:
        st.warning("No standard business features were recognized. You can still explore the detected columns manually.")

    total_cells = max(1, len(working) * len(working.columns))
    missing_cells = int(working.isna().sum().sum())
    missing_pct = missing_cells / total_cells * 100
    duplicate_rows = profile["duplicate_rows"]
    with st.expander("Data quality checks", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.metric("Missing cells", f"{missing_cells:,}", f"{missing_pct:.2f}% of all cells")
        d2.metric("Duplicate rows", f"{duplicate_rows:,}")
        d3.metric("Identifier fields", f"{len(profile['identifiers']):,}")
        high_missing = [(c, n) for c, n in profile["missing"].items() if n / max(1, len(working)) >= 0.20]
        if high_missing:
            st.warning("Columns with at least 20% missing values: " + ", ".join(f"{c} ({n:,})" for c, n in high_missing[:12]))
        else:
            st.success("No columns have 20% or more missing values.")
        quality_report = _build_data_quality_report(working, profile)
        st.markdown("**Column-level validation**")
        st.dataframe(quality_report, use_container_width=True, hide_index=True)
        invalid_total = int(quality_report["Additional issues"].sum())
        if invalid_total:
            st.warning(
                f"Found {invalid_total:,} potential parsing issues or statistical outliers. "
                "Outliers are flagged for review, not automatically treated as errors."
            )
        else:
            st.success("No additional parsing issues or IQR outliers were detected by these checks.")
        st.caption(
            "These checks are diagnostic only. RetailIQ does not automatically delete, impute, "
            "or modify source values. Statistical outliers may be valid business events."
        )

    st.subheader(f"Adaptive overview — {domain}")
    _render_domain_kpis(working, domain)
    overview = st.columns(3)
    overview[0].metric("Distinct categories", f"{sum(working[c].nunique(dropna=True) for c in category_options):,}")
    overview[1].metric("Numeric fields", f"{len(numeric_options):,}")
    overview[2].metric("Missing cells", f"{missing_cells:,}")

    if not numeric_options and not category_options and not date_options:
        st.info("No suitable numeric, date or categorical fields were detected.")
        st.dataframe(working.head(100), use_container_width=True)
        return

    # Choose a useful measure by semantic priority, excluding identifiers, flags and coordinates.
    preferred_measure = None
    measure_aliases = {
        "HR / Employee Analytics": ["MonthlyIncome", "Age", "YearsAtCompany"],
        "Banking / Customer Churn": ["Balance", "EstimatedSalary", "CreditScore", "Age"],
        "Supply Chain / Logistics": ["Sales", "Order Item Total", "Order Profit Per Order", "Days for shipping (real)"],
        "General Dataset Analytics": ["Sales", "Revenue", "Amount", "Profit", "Price"]
    }
    semantic_revenue = validated_roles.get("Revenue / Sales")
    semantic_quantity = validated_roles.get("Quantity / Units")
    semantic_price = validated_roles.get("Unit price")
    for candidate in [semantic_revenue, semantic_quantity, semantic_price]:
        if candidate in numeric_options:
            preferred_measure = candidate
            break
    if preferred_measure is None:
        for alias in measure_aliases.get(domain, []):
            found = _find_column(working, [alias])
            if found in numeric_options:
                preferred_measure = found
                break
    if preferred_measure is None and numeric_options:
        preferred_measure = numeric_options[0]

    st.subheader("Explore your data")
    measure = st.selectbox("Numeric measure", numeric_options,
                           index=numeric_options.index(preferred_measure) if preferred_measure in numeric_options else 0,
                           key=f"adaptive_measure_{_clean_name(base_name)}") if numeric_options else None
    if measure:
        values = pd.to_numeric(working[measure], errors="coerce").dropna()
        a1, a2, a3 = st.columns(3)
        a1.metric(f"Total {measure}", f"{values.sum():,.2f}" if len(values) else "—")
        a2.metric(f"Average {measure}", f"{values.mean():,.2f}" if len(values) else "—")
        a3.metric(f"Median {measure}", f"{values.median():,.2f}" if len(values) else "—")

    # Prefer meaningful domain category fields rather than stale widget selections.
    preferred_category = None
    category_aliases = {
        "HR / Employee Analytics": ["Attrition", "Department", "JobRole", "Gender"],
        "Banking / Customer Churn": ["Exited", "Geography", "Gender", "IsActiveMember"],
        "Supply Chain / Logistics": ["Delivery Status", "Order Region", "Type", "Shipping Mode", "Market"],
        "General Dataset Analytics": []
    }
    semantic_category_priority = [
        validated_roles.get("Category"),
        validated_roles.get("Product"),
        validated_roles.get("Geography"),
        validated_roles.get("Customer"),
        validated_roles.get("Status"),
    ]
    for candidate in semantic_category_priority:
        if candidate in category_options:
            preferred_category = candidate
            break
    if preferred_category is None:
        for alias in category_aliases.get(domain, []):
            found = _find_column(working, [alias])
            if found in category_options:
                preferred_category = found
                break
    if preferred_category is None and category_options:
        preferred_category = category_options[0]

    chart_recommendations = _recommend_chart_types(working, profile, validated_roles)
    if chart_recommendations:
        st.subheader("Recommended visualizations")
        st.caption("Chart suggestions are based on detected column roles and category counts. They are starting points, not proof of a business relationship.")
        rec_cols = st.columns(min(3, len(chart_recommendations)))
        for idx, (purpose, chart_type, detail) in enumerate(chart_recommendations):
            with rec_cols[idx % len(rec_cols)]:
                st.markdown(f"**{purpose}**")
                st.write(chart_type)
                st.caption(detail)

    tabs = st.tabs(["Overview & visuals", "Relationships", "Data preview"])
    with tabs[0]:
        if date_options:
            semantic_date = validated_roles.get("Date / Timestamp")
            date_default = semantic_date if semantic_date in date_options else date_options[0]
            date_col = st.selectbox("Date field", date_options, index=date_options.index(date_default),
                                    key=f"adaptive_date_{_clean_name(base_name)}")
            if measure:
                trend_agg = st.selectbox("Time trend aggregation", ["Sum", "Average", "Record count"],
                                         index=1 if any(t in _clean_name(measure) for t in ["days", "duration", "age", "score", "rate"]) else 0,
                                         key=f"adaptive_time_agg_{_clean_name(base_name)}")
                trend = _cached_monthly_trend(working, date_col, measure, trend_agg)
                if not trend.empty:
                    st.plotly_chart(px.line(trend, x=date_col, y=measure, markers=True,
                                            title=f"{trend_agg} {measure} by month", template="plotly_white"),
                                    use_container_width=True, key=f"adaptive_trend_{_clean_name(base_name)}")
                else:
                    st.caption("No valid date-and-measure pairs were found for this trend.")

        if category_options:
            category_col = st.selectbox("Category field", category_options,
                                        index=category_options.index(preferred_category) if preferred_category in category_options else 0,
                                        key=f"adaptive_category_{_clean_name(base_name)}")
            limit = st.slider("Number of categories to display", 5, 25, 10,
                              key=f"adaptive_category_limit_{_clean_name(base_name)}")
            chart_metric_options = ["Record count"]
            if measure:
                chart_metric_options += [f"Sum of {measure}", f"Average {measure}"]
            chart_metric = st.selectbox("Category chart value", chart_metric_options,
                                        key=f"adaptive_chart_metric_{_clean_name(base_name)}")
            if chart_metric == "Record count":
                grouped = working[category_col].fillna("Missing").value_counts().head(limit).rename("Records").reset_index()
                grouped.columns = [category_col, "Records"]
                y_col = "Records"
            else:
                temp = working[[category_col, measure]].copy()
                temp[measure] = pd.to_numeric(temp[measure], errors="coerce")
                grouped_obj = temp.dropna(subset=[measure]).groupby(category_col, dropna=False)[measure]
                grouped_obj = grouped_obj.sum() if chart_metric.startswith("Sum") else grouped_obj.mean()
                grouped = grouped_obj.sort_values(ascending=False).head(limit).reset_index()
                y_col = measure
            if not grouped.empty:
                title_metric = "Number of records" if chart_metric == "Record count" else chart_metric
                if len(grouped) > 8:
                    category_fig = px.bar(
                        grouped.sort_values(y_col), x=y_col, y=category_col,
                        orientation="h", title=f"{title_metric} by {category_col}",
                        template="plotly_white"
                    )
                else:
                    category_fig = px.bar(
                        grouped, x=category_col, y=y_col,
                        title=f"{title_metric} by {category_col}", template="plotly_white"
                    )
                st.plotly_chart(category_fig, use_container_width=True,
                                key=f"adaptive_category_bar_{_clean_name(base_name)}")
                if 2 <= len(grouped) <= 5:
                    st.plotly_chart(px.pie(grouped, names=category_col, values=y_col, hole=0.5,
                                           title=f"{title_metric} share by {category_col}"),
                                    use_container_width=True, key=f"adaptive_category_pie_{_clean_name(base_name)}")

        if measure:
            histogram_data = _cached_visual_sample(working, [measure])
            st.plotly_chart(px.histogram(histogram_data, x=measure, nbins=30,
                                         title=f"Distribution of {measure}", template="plotly_white"),
                            use_container_width=True, key=f"adaptive_hist_{_clean_name(base_name)}")
            if len(working) > 50000:
                st.caption("For responsiveness, this histogram uses a reproducible 50,000-row sample. KPIs and other full-data calculations still use all rows.")

        corr_fields = numeric_options[:15]
        if len(corr_fields) >= 2:
            corr = _cached_numeric_correlation(working, corr_fields)
            if corr.shape[0] >= 2:
                st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto",
                                          title="Correlation between numeric measures",
                                          color_continuous_scale="RdBu", zmin=-1, zmax=1),
                                use_container_width=True, key=f"adaptive_corr_{_clean_name(base_name)}")
                if len(working) > 50000:
                    st.caption("Correlation is calculated on a reproducible 50,000-row sample for faster rendering.")

        if not date_options and not category_options and not numeric_options:
            st.info("No chart is displayed because the data does not support a meaningful visual.")

    with tabs[1]:
        st.caption("Detected potential join keys, identifiers and geographic coordinates.")
        if profile["identifiers"]:
            st.write("Identifier fields:", ", ".join(map(str, profile["identifiers"])))
        if profile["geo"]:
            st.write("Geographic fields:", ", ".join(map(str, profile["geo"])))
        if len(table_names) > 1:
            st.write("Uploaded tables:", ", ".join(table_names))
            st.info("Review key meaning and row counts before relying on joined metrics.")
        else:
            st.info("Upload related tables together to explore possible relationships.")

    with tabs[2]:
        st.dataframe(working.head(500), use_container_width=True)
        st.download_button("Download analysed table", data=_cached_csv_export(working).encode("utf-8"),
                           file_name="retailiq_analysed_data.csv", mime="text/csv")


# Detect Olist multi-table upload by its characteristic file names.
olist_orders = find_uploaded_file(uploaded_data, "olist_orders_dataset")
olist_items = find_uploaded_file(uploaded_data, "olist_order_items_dataset")
olist_products = find_uploaded_file(uploaded_data, "olist_products_dataset")
olist_customers = find_uploaded_file(uploaded_data, "olist_customers_dataset")
olist_sellers = find_uploaded_file(uploaded_data, "olist_sellers_dataset")
olist_reviews = find_uploaded_file(uploaded_data, "olist_order_reviews_dataset")
olist_translation = find_uploaded_file(uploaded_data, "product_category_name_translation")

is_olist_upload = any([
    olist_orders is not None,
    olist_items is not None,
    olist_products is not None,
    olist_customers is not None
])

# Keep the original retail transaction dashboard unchanged for compatible
# single-file transaction datasets, and keep the existing Olist path.
# Other schemas use the adaptive analytics workspace instead of being forced
# into Online Retail's InvoiceNo/StockCode/Quantity/UnitPrice assumptions.
if not is_olist_upload and (
    len(uploaded_files) > 1
    or not _retail_schema_is_compatible(next(iter(uploaded_data.values())))
):
    _render_adaptive_workspace(uploaded_data)
    st.stop()

if is_olist_upload:
    # Orders + order items are the minimum reliable grain for Olist sales.
    # Product, customer, seller, review and translation files enrich analysis
    # but are not required to load the dataset.
    missing_olist = []
    if olist_orders is None:
        missing_olist.append("olist_orders_dataset.csv")
    if olist_items is None:
        missing_olist.append("olist_order_items_dataset.csv")

    if missing_olist:
        st.error(
            "RetailIQ needs the Olist orders and order-items files to calculate "
            "reliable sales and date-based metrics. Missing: " + ", ".join(missing_olist)
        )
        st.info(
            "Product, customer, seller, review and category-translation files are "
            "optional. Upload them when available to unlock additional analysis."
        )
        st.stop()

    required_schemas = {
        "orders": {"order_id", "customer_id", "order_purchase_timestamp", "order_status"},
        "items": {"order_id", "order_item_id", "product_id", "seller_id", "price"},
    }
    supplied_tables = {"orders": olist_orders, "items": olist_items}
    schema_errors = []
    for table_name, required in required_schemas.items():
        missing = required - set(supplied_tables[table_name].columns)
        if missing:
            schema_errors.append(f"{table_name}: {sorted(missing)}")
    if schema_errors:
        st.error("The uploaded Olist files have unexpected columns: " + "; ".join(schema_errors))
        st.stop()

    optional_status = {
        "Products": olist_products is not None,
        "Customers": olist_customers is not None,
        "Sellers": olist_sellers is not None,
        "Reviews": olist_reviews is not None,
        "Category translation": olist_translation is not None,
    }
    unavailable = [name for name, available in optional_status.items() if not available]
    if unavailable:
        st.info(
            "Partial Olist upload detected. Available data will be analysed; "
            "these optional features are unavailable until you upload: "
            + ", ".join(unavailable) + "."
        )

    orders_for_join = olist_orders.copy()
    items_for_join = olist_items.copy()
    orders_for_join["order_purchase_timestamp"] = pd.to_datetime(
        orders_for_join["order_purchase_timestamp"], errors="coerce"
    )
    orders_for_join = orders_for_join.dropna(subset=["order_id"])
    items_for_join = items_for_join.dropna(subset=["order_id", "product_id"])
    items_for_join["price"] = pd.to_numeric(items_for_join["price"], errors="coerce")
    items_for_join = items_for_join.dropna(subset=["price"])
    items_for_join = items_for_join[items_for_join["price"] > 0]

    delivered_orders = orders_for_join[
        orders_for_join["order_status"].astype(str).str.lower().eq("delivered")
    ].copy()
    if delivered_orders.empty:
        st.error("No delivered Olist orders were found. Completed-sales analysis cannot continue.")
        st.stop()

    combined = items_for_join.merge(
        delivered_orders[["order_id", "customer_id", "order_purchase_timestamp", "order_status"]],
        on="order_id", how="inner", validate="many_to_one"
    )

    # Optional product dimension: retain item rows even when product metadata is absent.
    if olist_products is not None and {"product_id", "product_category_name"}.issubset(olist_products.columns):
        products_for_join = olist_products.drop_duplicates(subset=["product_id"])
        combined = combined.merge(
            products_for_join, on="product_id", how="left", validate="many_to_one"
        )
        category_name = combined["product_category_name"].fillna("Uncategorised")
    else:
        # Without the optional products table, use product_id as a reliable
        # item-level label instead of grouping every item as "Uncategorised".
        combined["product_category_name"] = "Product ID"
        category_name = combined["product_id"].astype(str)

    # Optional customer dimension.
    if olist_customers is not None and {
        "customer_id", "customer_unique_id", "customer_city", "customer_state"
    }.issubset(olist_customers.columns):
        customers_for_join = olist_customers.drop_duplicates(subset=["customer_id"])
        combined = combined.merge(
            customers_for_join[[
                "customer_id", "customer_unique_id", "customer_city", "customer_state"
            ]],
            on="customer_id", how="left", validate="many_to_one"
        )
    else:
        combined["customer_unique_id"] = combined["customer_id"]
        combined["customer_city"] = np.nan
        combined["customer_state"] = np.nan

    if olist_translation is not None and {
        "product_category_name", "product_category_name_english"
    }.issubset(olist_translation.columns):
        translation = olist_translation.drop_duplicates(subset=["product_category_name"])
        combined = combined.merge(
            translation, on="product_category_name", how="left", validate="many_to_one"
        )
        category_name = combined["product_category_name_english"].fillna(
            combined["product_category_name"]
        )

    # Optional seller and review dimensions are joined without multiplying item rows.
    if olist_sellers is not None and {"seller_id", "seller_city", "seller_state"}.issubset(olist_sellers.columns):
        seller_dim = olist_sellers.drop_duplicates(subset=["seller_id"])
        combined = combined.merge(
            seller_dim[["seller_id", "seller_city", "seller_state"]],
            on="seller_id", how="left", validate="many_to_one"
        )

    if olist_reviews is not None and {"order_id", "review_score"}.issubset(olist_reviews.columns):
        review_summary = (
            olist_reviews.assign(review_score=pd.to_numeric(
                olist_reviews["review_score"], errors="coerce"
            ))
            .groupby("order_id", as_index=False)
            .agg(ReviewScore=("review_score", "mean"))
        )
        combined = combined.merge(
            review_summary, on="order_id", how="left", validate="many_to_one"
        )

    raw_df = pd.DataFrame({
        "InvoiceNo": combined["order_id"],
        "StockCode": combined["product_id"],
        "Description": category_name.fillna("Uncategorised"),
        "Quantity": 1,
        "InvoiceDate": combined["order_purchase_timestamp"],
        "UnitPrice": combined["price"],
        "Country": "Brazil",
        "CustomerID": combined["customer_unique_id"],
    })
    for optional_col in [
        "customer_city", "customer_state", "seller_id", "seller_city",
        "seller_state", "freight_value", "order_item_id", "ReviewScore",
        "product_category_name"
    ]:
        if optional_col in combined.columns:
            raw_df[optional_col] = combined[optional_col].values

    st.success(
        f"Olist data integrated: {len(raw_df):,} order-item rows "
        f"from {raw_df['InvoiceNo'].nunique():,} delivered orders."
    )
    st.caption(
        "Olist preparation: one row per order item; Quantity is represented as 1 per item row; "
        "revenue uses item price and excludes freight. Only delivered orders are included. "
        "When the products file is missing, product_id is used as the product label; "
        "category-level insights require the optional products file. "
        "Customer/seller/review insights depend on the optional files uploaded."
    )

else:
    if len(uploaded_files) != 1:
        st.error(
            "These files were not recognised as an Olist multi-file dataset. "
            "Please upload one standard transaction file, or the four required Olist files."
        )
        st.stop()

    raw_df = next(iter(uploaded_data.values()))

    required_columns = [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice"
    ]
    column_aliases = {
        "InvoiceNo": ["invoiceno", "invoice", "invoiceid", "orderid", "ordernumber", "transactionid"],
        "StockCode": ["stockcode", "productid", "productcode", "sku", "itemcode"],
        "Description": ["description", "productname", "itemname", "product", "productcategoryname"],
        "Quantity": ["quantity", "qty", "unitssold", "numberofitems"],
        "InvoiceDate": ["invoicedate", "orderdate", "transactiondate", "date", "orderpurchasetimestamp"],
        "UnitPrice": ["unitprice", "sellingprice", "priceperunit", "price"],
        "Country": ["country", "customercountry", "market", "salescountry"],
        "CustomerID": ["customerid", "customeruniqueid", "clientid", "buyerid"],
    }
    column_lookup = {normalize_column(col): col for col in raw_df.columns}
    rename_map = {}
    used_columns = set()
    unmapped_columns = []

    for target in required_columns + ["Country", "CustomerID"]:
        matched = None
        for alias in column_aliases[target]:
            candidate = column_lookup.get(normalize_column(alias))
            if candidate is not None and candidate not in used_columns:
                matched = candidate
                break
        if matched is not None:
            rename_map[matched] = target
            used_columns.add(matched)
        elif target in required_columns:
            unmapped_columns.append(target)

    if unmapped_columns:
        st.subheader("A few columns need your attention")
        st.info(
            "RetailIQ AI could not confidently identify every required field. "
            "Select the matching columns below."
        )
        for target in unmapped_columns:
            available = [c for c in raw_df.columns if c not in used_columns]
            selected = st.selectbox(
                f"Select column for {target}",
                ["-- Select a column --"] + available,
                key=f"mapping_{target}"
            )
            if selected != "-- Select a column --":
                rename_map[selected] = target
                used_columns.add(selected)

    mapped_targets = list(rename_map.values())
    missing = [c for c in required_columns if c not in mapped_targets]
    if missing:
        st.warning(f"Please map these required fields: {missing}")
        st.stop()

    raw_df = raw_df.rename(columns=rename_map)
    if "Country" not in raw_df.columns:
        raw_df["Country"] = "Unknown"
        st.info("No country column was detected. Market analysis will show 'Unknown'.")
    if "CustomerID" not in raw_df.columns:
        raw_df["CustomerID"] = np.nan

    st.success("Dataset columns successfully mapped!")
    with st.expander("Review automatically detected columns"):
        mapping_df = pd.DataFrame(
            list(rename_map.items()),
            columns=["Original Column", "RetailIQ AI Column"]
        )
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)


# ==========================================
# DATA CLEANING
# ==========================================

df = raw_df.copy()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
df["Country"] = df["Country"].fillna("Unknown").astype(str)
df["Description"] = df["Description"].fillna("Uncategorised").astype(str)

cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
returns = df["Quantity"] < 0
return_count = int(returns.sum())
cancelled_count = int(cancelled.sum())

df = df[
    (~cancelled) &
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0) &
    (df["InvoiceDate"].notna()) &
    (df["StockCode"].notna())
].copy()

df["Revenue"] = df["Quantity"] * df["UnitPrice"]

if df.empty:
    st.error("No valid sales records found after data cleaning.")
    st.stop()


# ==========================================
# BUSINESS METRICS
# ==========================================

total_revenue = df["Revenue"].sum()

total_orders = df["InvoiceNo"].nunique()

total_units = df["Quantity"].sum()

total_products = df["StockCode"].nunique()

total_customers = (
    df["CustomerID"].nunique()
    if "CustomerID" in df.columns
    else 0
)

aov = (
    total_revenue / total_orders
    if total_orders else 0
)

# ==========================================
# ANALYSIS TABLES
# ==========================================

df["Month"] = (
    df["InvoiceDate"].dt.to_period("M")
    .dt.to_timestamp()
)

df["Date"] = (
    df["InvoiceDate"].dt.normalize()
)

monthly = (
    df.groupby("Month")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Units=("Quantity", "sum")
    )
    .reset_index()
)

products = (
    df.groupby("Description")
    .agg(
        Revenue=("Revenue", "sum"),
        Units=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique")
    )
    .sort_values("Revenue", ascending=False)
    .reset_index()
)

countries = (
    df.groupby("Country")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique")
    )
    .sort_values("Revenue", ascending=False)
    .reset_index()
)

daily = (
    df.groupby("Date")["Revenue"]
    .sum()
    .sort_index()
)

date_range = pd.date_range(
    daily.index.min(),
    daily.index.max(),
    freq="D"
)

daily = daily.reindex(
    date_range, fill_value=0
)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

page_options = [
        "Home",
        "Business Overview",
        "Sales Analytics",
        "Product Intelligence",
        "Market Analysis",
        "Sales Forecast",
        "AI Business Advisor"
    ]
page = st.sidebar.radio(
    "Navigation",
    page_options,
    index=page_options.index(st.session_state.get("retailiq_selected_page", "Home"))
    if st.session_state.get("retailiq_selected_page", "Home") in page_options else 0,
    key="retailiq_navigation"
)
st.session_state["retailiq_selected_page"] = page

page_descriptions = {
    "Home": "Your retail intelligence workspace at a glance.",
    "Business Overview": "A high-level view of revenue, orders, customers and overall performance.",
    "Sales Analytics": "Explore sales trends, order patterns and revenue movement.",
    "Product Intelligence": "Understand product contribution, demand and performance.",
    "Market Analysis": "Compare customer and geographic market performance.",
    "Sales Forecast": "Review historical patterns and model-based sales projections.",
    "AI Business Advisor": "Turn business signals into practical recommendations and actions.",
}
st.markdown('<div class="workspace-kicker">RETAILIQ AI  /  BUSINESS INTELLIGENCE</div>',
            unsafe_allow_html=True)
st.title(page)
st.markdown(
    f'<div class="workspace-subtitle">{page_descriptions.get(page, "Explore your retail data.")}</div>',
    unsafe_allow_html=True
)
st.divider()

# ==========================================
# HOME: RETAILIQ AI WORKSPACE
# ==========================================

if page == "Home":
    home_left, home_right = st.columns([1.65, 0.85], gap="large")
    with home_left:
        st.markdown(
            '<div class="ri-kicker">✦ AI RETAIL ANALYTICS PLATFORM</div>'
            '<div class="ri-hero-title">Turn Your Retail Data<br>Into <span>Powerful Insights</span></div>'
            '<div class="ri-subtitle">Explore your retail performance, discover customer behaviour, '
            'track product and market trends, and make smarter decisions with RetailIQ AI.</div>',
            unsafe_allow_html=True
        )
    with home_right:
        st.markdown('<div class="ri-orb-wrap"><div class="ri-orb"></div></div>', unsafe_allow_html=True)

    st.markdown("### Explore what you can do")
    home_modules = [
        ("▥", "Business Overview", "Key metrics and performance", "Business Overview"),
        ("↗", "Sales Analytics", "Revenue, orders and growth", "Sales Analytics"),
        ("⬡", "Product Intelligence", "Products and categories", "Product Intelligence"),
        ("⌖", "Market Analysis", "Markets and geography", "Market Analysis"),
        ("⌁", "Sales Forecast", "Future sales projections", "Sales Forecast"),
        ("✧", "AI Business Advisor", "Insights and recommendations", "AI Business Advisor")
    ]
    home_cols = st.columns(6)
    for idx, (icon_text, title_text, copy_text, page_name) in enumerate(home_modules):
        with home_cols[idx]:
            st.markdown(
                f'<div class="ri-glass ri-module"><div class="ri-module-icon">{icon_text}</div>'
                f'<div class="ri-module-title">{title_text}</div>'
                f'<div class="ri-module-copy">{copy_text}</div></div>',
                unsafe_allow_html=True
            )
            if st.button("Open →", key=f"home_open_{idx}", use_container_width=True):
                st.session_state["retailiq_selected_page"] = page_name
                st.session_state["retailiq_navigation"] = page_name
                st.rerun()

    st.markdown("### Your business at a glance")
    summary_col, files_col = st.columns([1.45, 1], gap="large")
    with summary_col:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Revenue", f"{currency_symbol}{total_revenue:,.0f}")
        k2.metric("Total Orders", f"{total_orders:,}")
        k3.metric("Customers", f"{total_customers:,}")
        k4.metric("Products", f"{total_products:,}")
        st.caption("Live metrics calculated from the currently uploaded and processed dataset.")
    with files_col:
        st.markdown('<div class="ri-glass" style="padding:1rem 1.1rem;">'
                    '<div class="ri-panel-title">✦ Your data workspace</div>'
                    '<div class="ri-mini-row"><span>Uploaded tables</span><b>' +
                    f'{len(uploaded_data):,}' +
                    '</b></div><div class="ri-mini-row"><span>Total uploaded rows</span><b>' +
                    f'{uploaded_table_rows:,}' +
                    '</b></div><div class="ri-mini-row"><span>Data status</span><b class="ri-ready">Ready</b></div>'
                    '</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="ri-panel-title">Recent datasets</div>', unsafe_allow_html=True)
        for table_name, table_frame in list(uploaded_data.items())[:4]:
            st.markdown(
                f'<div class="ri-mini-row"><span>▤ &nbsp; {table_name}</span>'
                f'<b>{len(table_frame):,} rows</b></div>',
                unsafe_allow_html=True
            )

# ==========================================
# PAGE 1: BUSINESS OVERVIEW
# ==========================================

if page == "Business Overview":

    st.header("Business Performance Overview")

    # ==========================================
    # KPI CARDS
    # ==========================================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Revenue",
        f"{currency_symbol}{total_revenue:,.2f}"
    )

    c2.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    c3.metric(
        "Average Order Value",
        f"{currency_symbol}{aov:,.2f}"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric("Units Sold", f"{total_units:,.0f}")
    c5.metric("Unique Products", f"{total_products:,}")
    c6.metric("Unique Customers", f"{total_customers:,}")

    # ==========================================
    # CHART THEME
    # ==========================================

    navy = "#14213D"
    teal = "#14B8A6"
    light_blue = "#DCE6F1"

    chart_colors = [
        navy, teal, "#38BDF8", "#64748B",
        "#A5B4FC", "#2DD4BF", "#94A3B8",
        "#0F766E"
    ]

    # ==========================================
    # MONTHLY REVENUE TREND
    # ==========================================

    st.subheader("Monthly Revenue Trend")

    fig_monthly = px.line(
        monthly,
        x="Month",
        y="Revenue",
        markers=True,
        template="plotly_white"
    )

    fig_monthly.update_traces(
        line=dict(color=teal, width=3),
        marker=dict(size=7),
        hovertemplate=(
            "Month: %{x|%B %Y}<br>"
            f"Revenue: {currency_symbol}%{{y:,.2f}}"
            "<extra></extra>"
        )
    )

    fig_monthly.update_layout(
        title="Monthly Revenue Performance",
        title_font_color=navy,
        yaxis_title=f"Revenue ({selected_currency})",
        xaxis_title="Month",
        yaxis_tickprefix=currency_symbol,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

    # ==========================================
    # COUNTRY REVENUE DONUT — DATA-DRIVEN VISIBILITY
    # ==========================================

    if countries["Country"].nunique() > 1:
        st.subheader("Revenue Distribution by Country")

        country_chart = countries.head(8).copy()

        if len(countries) > 8:
            other_revenue = countries.iloc[8:]["Revenue"].sum()
            country_chart = pd.concat(
                [
                    country_chart,
                    pd.DataFrame({
                        "Country": ["Other Countries"],
                        "Revenue": [other_revenue]
                    })
                ],
                ignore_index=True
            )

        fig_donut = px.pie(
            country_chart,
            names="Country",
            values="Revenue",
            hole=0.58,
            color_discrete_sequence=chart_colors
        )

        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate=(
                "Country: %{label}<br>"
                f"Revenue: {currency_symbol}%{{value:,.2f}}"
                "<br>Share: %{percent}"
                "<extra></extra>"
            )
        )

        fig_donut.update_layout(
            title="Country-wise Revenue Share",
            title_font_color=navy,
            font=dict(color=navy),
            legend_title="Markets",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.caption(
            "Country-share chart hidden: only one country is present, "
            "so a country comparison would not add useful information."
        )

    # ==========================================
    # DAILY SALES TREND
    # ==========================================

    st.subheader("Daily Sales Performance")

    daily_chart = daily.rename_axis(
        "Date"
    ).reset_index(name="Revenue")

    fig_daily = px.line(
        daily_chart,
        x="Date",
        y="Revenue",
        template="plotly_white"
    )

    fig_daily.update_traces(
        line=dict(color=teal, width=2),
        hovertemplate=(
            "Date: %{x|%d %b %Y}<br>"
            f"Revenue: {currency_symbol}%{{y:,.2f}}"
            "<extra></extra>"
        )
    )

    fig_daily.update_layout(
        title="Daily Revenue Trend",
        title_font_color=navy,
        xaxis_title="Date",
        yaxis_title=f"Revenue ({selected_currency})",
        yaxis_tickprefix=currency_symbol,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True
    )


    # ==========================================
    # ORDER VALUE DISTRIBUTION
    # ==========================================

    st.subheader("Order Value Distribution")

    order_values = (
        df.groupby("InvoiceNo")["Revenue"]
        .sum()
        .reset_index(name="OrderValue")
    )

    # Keep only positive order values
    positive_orders = order_values[
        order_values["OrderValue"] > 0
    ]["OrderValue"]

    if not positive_orders.empty:

        # Summary metrics
        avg_order = positive_orders.mean()
        median_order = positive_orders.median()
        max_order = positive_orders.max()

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Average Order Value",
            f"{currency_symbol}{avg_order:,.2f}"
        )

        m2.metric(
            "Median Order Value",
            f"{currency_symbol}{median_order:,.2f}"
        )

        m3.metric(
            "Highest Order Value",
            f"{currency_symbol}{max_order:,.2f}"
        )

        # Create logarithmic bins manually
        min_value = positive_orders.min()
        max_value = positive_orders.max()

        if min_value == max_value:
            bin_edges = np.linspace(
                min_value * 0.5,
                max_value * 1.5,
                11
            )
        else:
            bin_edges = np.geomspace(
                min_value,
                max_value * 1.001,
                21
            )

        counts, edges = np.histogram(
            positive_orders,
            bins=bin_edges
        )

        histogram_df = pd.DataFrame({
            "Order Range": [
                f"{currency_symbol}{edges[i]:,.0f} - "
                f"{currency_symbol}{edges[i+1]:,.0f}"
                for i in range(len(counts))
            ],
            "Number of Orders": counts
        })

        histogram_df = histogram_df[
            histogram_df["Number of Orders"] > 0
        ]

        fig_hist = px.bar(
            histogram_df,
            x="Order Range",
            y="Number of Orders",
            title="Order Value Distribution",
            color_discrete_sequence=["#14B8A6"],
            template="plotly_white"
        )

        fig_hist.update_traces(
            marker_line_color="#14213D",
            marker_line_width=0.5,
            hovertemplate=(
                "Order Range: %{x}<br>"
                "Number of Orders: %{y:,}"
                "<extra></extra>"
            )
        )

        fig_hist.update_layout(
            title_font_color="#14213D",
            xaxis_title="Order Value Ranges",
            yaxis_title="Number of Orders",
            xaxis_tickangle=-35,
            font=dict(color="#14213D"),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    else:
        st.warning("No positive order values available.")



    # ==========================================
    # GLOBAL SALES MAP
    # ==========================================
    if countries["Country"].nunique() > 1:
        st.subheader("Global Sales Map")

        # Convert country names into standard ISO-3 codes
        country_iso3 = {
            "United Kingdom": "GBR",
            "Netherlands": "NLD",
            "EIRE": "IRL",
            "Germany": "DEU",
            "France": "FRA",
            "Australia": "AUS",
            "Spain": "ESP",
            "Switzerland": "CHE",
            "Belgium": "BEL",
            "Sweden": "SWE",
            "Italy": "ITA",
            "Portugal": "PRT",
            "Norway": "NOR",
            "Finland": "FIN",
            "Denmark": "DNK",
            "Austria": "AUT",
            "Cyprus": "CYP",
            "Greece": "GRC",
            "Poland": "POL",
            "Japan": "JPN",
            "Singapore": "SGP",
            "Hong Kong": "HKG",
            "USA": "USA",
            "Canada": "CAN",
            "Brazil": "BRA",
            "Israel": "ISR",
            "Iceland": "ISL",
            "Bahrain": "BHR",
            "Lebanon": "LBN",
            "Lithuania": "LTU",
            "Malta": "MLT",
            "Saudi Arabia": "SAU",
            "United Arab Emirates": "ARE",
            "Czech Republic": "CZE",
            "RSA": "ZAF"
        }

        map_countries = countries.copy()

        map_countries["ISO3"] = (
            map_countries["Country"].map(country_iso3)
        )

        map_countries = map_countries.dropna(
            subset=["ISO3"]
        )

        if not map_countries.empty:

            fig_map = px.choropleth(
                map_countries,
                locations="ISO3",
                locationmode="ISO-3",
                color="Revenue",
                hover_name="Country",
                color_continuous_scale=[
                    "#DCE6F1",
                    "#14B8A6",
                    "#14213D"
                ],
                template="plotly_white"
            )

            fig_map.update_geos(
                fitbounds="locations",
                visible=True,
                showcountries=True,
                showcoastlines=True,
                showland=True,
                landcolor="#E2E8F0",
                bgcolor="white"
            )

            fig_map.update_layout(
                title="Worldwide Revenue Distribution",
                coloraxis_colorbar=dict(
                    title=f"Revenue ({selected_currency})"
                ),
                paper_bgcolor="white",
                font=dict(color="#14213D")
            )

            fig_map.update_traces(
                hovertemplate=(
                    "Country: %{hovertext}<br>"
                    f"Revenue: {currency_symbol}%{{z:,.2f}}"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(
                fig_map,
                use_container_width=True
            )

        else:
            st.warning("No recognized country locations found.")


    else:
        st.caption(
            "Global country map hidden: the dataset contains only one country. "
            "A geographic comparison will appear when multiple countries are available."
        )


    # ==========================================
    # TOP PERFORMING PRODUCTS
    # ==========================================

    st.subheader("Top Performing Products")

    display_products = products.head(10).copy()

    display_products["Revenue"] = (
        display_products["Revenue"].apply(
            lambda x: f"{currency_symbol}{x:,.2f}"
        )
    )

    st.dataframe(
        display_products,
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# PAGE 2: SALES ANALYTICS
# ==========================================


elif page == "Sales Analytics":

    st.header("Sales Analytics")
    navy = "#14213D"
    teal = "#14B8A6"

    # ==========================================
    # MONTHLY REVENUE
    # ==========================================

    st.subheader("Monthly Revenue Performance")

    fig = px.bar(
        monthly,
        x="Month",
        y="Revenue",
        title="Monthly Revenue",
        color_discrete_sequence=[teal],
        template="plotly_white"
    )

    fig.update_layout(
        yaxis_title=f"Revenue ({selected_currency})",
        xaxis_title="Month",
        yaxis_tickprefix=currency_symbol,
        title_font_color=navy,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.update_traces(
        hovertemplate=(
            "Month: %{x|%B %Y}<br>"
            f"Revenue: {currency_symbol}%{{y:,.2f}}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # MONTHLY ORDERS
    # ==========================================

    st.subheader("Monthly Order Trends")

    fig2 = px.line(
        monthly,
        x="Month",
        y="Orders",
        markers=True,
        title="Monthly Orders",
        color_discrete_sequence=[navy],
        template="plotly_white"
    )

    fig2.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Orders",
        title_font_color=navy,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # DAILY SALES WITH DATE FILTER
    # ==========================================

    st.subheader("Daily Sales Analysis")

    min_date = df["InvoiceDate"].min().date()
    max_date = df["InvoiceDate"].max().date()

    selected_dates = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

        start_date, end_date = selected_dates

        filtered_df = df[
            df["InvoiceDate"].dt.date.between(
                start_date, end_date
            )
        ].copy()

        if not filtered_df.empty:

            filtered_daily = (
                filtered_df.groupby(
                    filtered_df["InvoiceDate"].dt.date
                )["Revenue"]
                .sum()
                .reset_index()
            )

            filtered_daily.columns = ["Date", "Revenue"]

            fig3 = px.area(
                filtered_daily,
                x="Date",
                y="Revenue",
                title="Daily Revenue Trend",
                color_discrete_sequence=[teal],
                template="plotly_white"
            )

            fig3.update_layout(
                xaxis_title="Date",
                yaxis_title=f"Revenue ({selected_currency})",
                yaxis_tickprefix=currency_symbol,
                title_font_color=navy,
                font=dict(color=navy),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

        else:
            st.warning("No sales found in this date range.")

    # ==========================================
    # WEEKDAY REVENUE ANALYSIS
    # ==========================================

    st.subheader("Revenue by Day of the Week")

    weekday_df = df.copy()

    weekday_df["Weekday"] = (
        weekday_df["InvoiceDate"].dt.day_name()
    )

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    weekday_sales = (
        weekday_df.groupby("Weekday")["Revenue"]
        .sum()
        .reindex(weekday_order, fill_value=0)
        .reset_index()
    )

    fig4 = px.bar(
        weekday_sales,
        x="Weekday",
        y="Revenue",
        title="Which Days Generate the Most Revenue?",
        color_discrete_sequence=[teal],
        template="plotly_white"
    )

    fig4.update_layout(
        xaxis_title="Day of the Week",
        yaxis_title=f"Revenue ({selected_currency})",
        yaxis_tickprefix=currency_symbol,
        title_font_color=navy,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # ==========================================
    # MONTHLY PERFORMANCE TABLE
    # ==========================================

    st.subheader("Monthly Performance Data")

    display_monthly = monthly.copy()

    display_monthly["Revenue"] = (
        display_monthly["Revenue"].apply(
            lambda x: f"{currency_symbol}{x:,.2f}"
        )
    )

    st.dataframe(
        display_monthly,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# PAGE 3: PRODUCT INTELLIGENCE
# ==========================================


elif page == "Product Intelligence":

    st.header("Product Intelligence")

    navy = "#14213D"
    teal = "#14B8A6"

    # ==========================================
    # TOP PRODUCTS BY REVENUE
    # ==========================================

    st.subheader("Top 10 Products by Revenue")

    top_products = products.head(10).copy()

    # Olist product IDs are long UUID-like values, not human-readable names.
    # Use a compact display label in charts while retaining the full ID for hover/table.
    product_ids_as_labels = bool(
        is_olist_upload
        and raw_df["Description"].astype(str).reset_index(drop=True).equals(
            raw_df["StockCode"].astype(str).reset_index(drop=True)
        )
    )
    if product_ids_as_labels:
        products["Product ID"] = products["Description"].astype(str)
        products["Display Product"] = products["Product ID"].apply(
            lambda value: f"Product · {value[:10]}…"
        )
        top_products = products.head(10).copy()
        chart_product_col = "Display Product"
    else:
        chart_product_col = "Description"

    fig = px.bar(
        top_products.sort_values("Revenue"),
        x="Revenue",
        y=chart_product_col,
        orientation="h",
        color="Revenue",
        color_continuous_scale=["#14B8A6", "#14213D"],
        template="plotly_white"
    )

    fig.update_layout(
        title="Highest Revenue Generating Products",
        xaxis_title=f"Revenue ({selected_currency})",
        xaxis_tickprefix=currency_symbol,
        yaxis_title="Product",
        title_font_color=navy,
        font=dict(color=navy),
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.update_traces(
        hovertemplate=(
            "Product: %{y}<br>"
            f"Revenue: {currency_symbol}%{{x:,.2f}}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # PRODUCT REVENUE DONUT
    # ==========================================

    st.subheader("Product Revenue Contribution")

    product_donut = products.head(8).copy()
    donut_label_col = chart_product_col

    if len(products) > 8:

        other_revenue = products.iloc[8:]["Revenue"].sum()

        product_donut = pd.concat(
            [
                product_donut,
                pd.DataFrame({
                    "Description": ["Other Products"],
                    "Revenue": [other_revenue],
                    **({"Display Product": ["Other Products"]} if product_ids_as_labels else {})
                })
            ],
            ignore_index=True
        )

    fig2 = px.pie(
        product_donut,
        names=donut_label_col,
        values="Revenue",
        hole=0.58,
        color_discrete_sequence=[
            navy, teal, "#38BDF8", "#64748B",
            "#A5B4FC", "#2DD4BF", "#94A3B8",
            "#0F766E", "#CBD5E1"
        ]
    )

    fig2.update_traces(
        textinfo="percent",
        hovertemplate=(
            "Product: %{label}<br>"
            f"Revenue: {currency_symbol}%{{value:,.2f}}"
            "<br>Share: %{percent}"
            "<extra></extra>"
        )
    )

    fig2.update_layout(
        title="Revenue Share by Product",
        title_font_color=navy,
        font=dict(color=navy),
        paper_bgcolor="white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # TOP PRODUCTS BY QUANTITY
    # ==========================================

    st.subheader("Top 10 Products by Quantity Sold")

    quantity_top = products.nlargest(
        10, "Units"
    ).sort_values("Units")

    fig3 = px.bar(
        quantity_top,
        x="Units",
        y=chart_product_col,
        orientation="h",
        color_discrete_sequence=[teal],
        template="plotly_white"
    )

    fig3.update_layout(
        title="Best Selling Products by Quantity",
        xaxis_title="Units Sold",
        yaxis_title="Product",
        title_font_color=navy,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # ==========================================
    # PRODUCT PERFORMANCE TABLE
    # ==========================================

    st.subheader("Complete Product Performance")

    display_products = products.copy()
    if product_ids_as_labels:
        display_products = display_products.rename(
            columns={"Display Product": "Product"}
        )
        display_products = display_products[
            ["Product", "Product ID", "Revenue", "Units", "Orders"]
        ]
    else:
        display_products = display_products.rename(
            columns={"Description": "Product"}
        )

    display_products["Revenue"] = (
        display_products["Revenue"].apply(
            lambda x: f"{currency_symbol}{x:,.2f}"
        )
    )

    st.dataframe(
        display_products,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# PAGE 4: MARKET ANALYSIS
# ==========================================



elif page == "Market Analysis":

    st.header("Market Analysis")

    country_count = countries["Country"].nunique()

    if country_count > 1:
        st.subheader("Country-wise Revenue Comparison")
        fig = px.bar(
            countries.head(10),
            x="Country",
            y="Revenue",
            title="Top Countries by Revenue",
            template="plotly_white"
        )
        fig.update_layout(
            yaxis_title=f"Revenue ({selected_currency})",
            yaxis_tickprefix=currency_symbol
        )
        fig.update_traces(
            hovertemplate=(
                "Country: %{x}<br>"
                f"Revenue: {currency_symbol}%{{y:,.2f}}"
                "<extra></extra>"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        display_countries = countries.copy()
        display_countries["Revenue"] = display_countries["Revenue"].apply(
            lambda x: f"{currency_symbol}{x:,.2f}"
        )
        st.dataframe(display_countries, use_container_width=True, hide_index=True)

        # Only show this legacy KPI when the selected data actually contains UK.
        uk_rows = countries[
            countries["Country"].astype(str).str.casefold() == "united kingdom"
        ]
        if not uk_rows.empty and total_revenue:
            uk_revenue = uk_rows["Revenue"].sum()
            st.metric(
                "United Kingdom Revenue Contribution",
                f"{uk_revenue / total_revenue * 100:.1f}%"
            )
    elif country_count == 1:
        only_market = countries.iloc[0]
        st.subheader("Available Market")
        m1, m2 = st.columns(2)
        m1.metric("Detected Country", str(only_market["Country"]))
        m2.metric("Revenue", f"{currency_symbol}{only_market['Revenue']:,.2f}")
        st.metric("Orders", f"{int(only_market['Orders']):,}")
        st.caption(
            "Country comparison charts and country-share metrics are hidden "
            "because only one country is present in this dataset."
        )
        display_countries = countries.copy()
        display_countries["Revenue"] = display_countries["Revenue"].apply(
            lambda x: f"{currency_symbol}{x:,.2f}"
        )
        st.dataframe(display_countries, use_container_width=True, hide_index=True)
    else:
        st.info("No country values are available for market analysis.")


# ==========================================
# PAGE 5: SALES FORECAST
# ==========================================

elif page == "Sales Forecast":

    st.header("Sales Forecasting")

    if len(daily) < 28:

        st.warning(
            "At least 28 days of sales data is required "
            "for this forecasting model."
        )

    else:

        try:

            seasonal = len(daily) >= 60

            model = ExponentialSmoothing(
                daily,
                trend="add",
                seasonal="add" if seasonal else None,
                seasonal_periods=7 if seasonal else None,
                initialization_method="estimated"
            )

            fitted = model.fit()

            # Hold out the most recent observations to report historical forecast error.
            backtest, backtest_note = _evaluate_time_series_forecast(daily)
            if backtest:
                eval_cols = st.columns(3)
                eval_cols[0].metric("Holdout MAE", f"{backtest['mae']:,.2f}")
                eval_cols[1].metric("Holdout RMSE", f"{backtest['rmse']:,.2f}")
                eval_cols[2].metric("Holdout MAPE", f"{backtest['mape']:.1f}%" if np.isfinite(backtest["mape"]) else "—")
                st.caption(backtest_note)
                comparison = pd.DataFrame({
                    "Date": backtest["actual"].index,
                    "Actual Revenue": backtest["actual"].values,
                    "Predicted Revenue": backtest["predicted"].values,
                })
                st.plotly_chart(
                    px.line(comparison, x="Date", y=["Actual Revenue", "Predicted Revenue"],
                            title="Forecast backtest: actual vs predicted", template="plotly_white"),
                    use_container_width=True
                )

            forecast = fitted.forecast(30)

            forecast_df = pd.DataFrame({
                "Date": forecast.index,
                "Predicted Revenue": forecast.values
            })

            
            st.metric(
                "Next 30-Day Forecast",
                f"{currency_symbol}{forecast.sum():,.2f}"
            )
            
            fig = px.line(
                forecast_df,
                x="Date",
                y="Predicted Revenue",
                markers=True,
                title="Future Revenue Forecast",
                template="plotly_white"
            )
            
            fig.update_layout(
                yaxis_title=f"Predicted Revenue ({selected_currency})",
                yaxis_tickprefix=currency_symbol
            )
            
            fig.update_traces(
                hovertemplate=
                f"Date: %{{x}}<br>"
                f"Predicted Revenue: {currency_symbol}%{{y:,.2f}}"
                "<extra></extra>"
            )

            st.plotly_chart(
                fig, use_container_width=True
            )

           
            st.dataframe(
                forecast_df.style.format({
                    "Predicted Revenue": f"{currency_symbol}{{:,.2f}}"
                }),
                use_container_width=True,
                hide_index=True
            )


        except Exception as e:
            st.error(f"Forecasting error: {e}")

# ==========================================
# PAGE 6: AI BUSINESS ADVISOR
# ==========================================

elif page == "AI Business Advisor":

    st.header("✨ AI Business Advisor")

    best_month = monthly.loc[
        monthly["Revenue"].idxmax()
    ]

    best_country = countries.iloc[0]
    best_product = products.iloc[0]

    st.subheader("Automated Business Insights")

    st.info(
        f"Your highest-revenue month was "
        f"{best_month['Month'].strftime('%B %Y')} "
        f"with {currency_symbol}{best_month['Revenue']:,.2f}."
    )

    st.info(
        f"Your leading market is {best_country['Country']} "
        f"with {currency_symbol}{best_country['Revenue']:,.2f} revenue."
    )

    st.info(
        f"Your highest-revenue product is "
        f"{best_product['Description']}."
    )

    if len(daily) >= 28:

        st.success(
            "Your sales forecasting feature is available. "
            "Use the Sales Forecast page to inspect predictions."
        )

    st.subheader("Ask RetailIQ")

    question = st.text_input(
        "Ask a question about your business:"
    )

    if st.button("Generate Answer") and question:

        q = question.lower()

        if "country" in q or "market" in q:

            answer = (
                f"{best_country['Country']} leads with "
                f"{currency_symbol}{best_country['Revenue']:,.2f} revenue."
            )

        elif "product" in q:

            answer = (
                f"{best_product['Description']} is the "
                f"highest-revenue product with "
                f"{currency_symbol}{best_product['Revenue']:,.2f} revenue."
            )

        elif "month" in q:

            answer = (
                f"{best_month['Month'].strftime('%B %Y')} "
                f"was the best month, generating "
                f"{currency_symbol}{best_month['Revenue']:,.2f}."
            )

        elif "customer" in q:

            answer = (
                f"The dataset contains {total_customers:,} "
                "identified customers."
            )

        elif "order" in q:

            answer = f"Total valid orders: {total_orders:,}."

        elif "sales" in q or "revenue" in q:

            answer = (
                f"Total revenue is {currency_symbol}{total_revenue:,.2f}."
            )

        else:

            answer = (
                "Try asking about revenue, sales, products, "
                "countries, customers or monthly performance."
            )

        st.success(answer)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "RetailIQ AI | Smart Retail Analytics & "
    "Business Decision Support | Portfolio MVP"
)
