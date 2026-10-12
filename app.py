import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from services.parser     import parse_resume
from services.embedder   import compute_similarity
from services.skills     import extract_skills
from services.matcher    import get_missing_skills
from services.suggester  import generate_suggestions
from services.analyzer   import (
    compute_ats_score, extract_experience_years,
    extract_github, extract_email, extract_phone,
    compute_keyword_density,
)
from services.exporter   import to_json, to_csv, to_pdf_report

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeAI — Intelligent Screening",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# MASTER CSS — Neon Purple × Black × White
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(r"""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── CSS Variables ── */
:root {
  --neon:      #b026ff;
  --neon-soft: #c96cff;
  --neon-glow: rgba(176,38,255,0.35);
  --neon-dim:  rgba(176,38,255,0.12);
  --bg:        #05050d;
  --bg2:       #0a0a18;
  --bg3:       #0f0f22;
  --border:    rgba(176,38,255,0.18);
  --border-hi: rgba(176,38,255,0.5);
  --white:     #ffffff;
  --white80:   rgba(255,255,255,0.80);
  --white50:   rgba(255,255,255,0.50);
  --white20:   rgba(255,255,255,0.20);
  --white08:   rgba(255,255,255,0.08);
  --white04:   rgba(255,255,255,0.04);
  --green:     #00f5a0;
  --amber:     #ffbe0b;
  --red:       #ff4d6d;
  --mono:      'JetBrains Mono', monospace;
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif !important;
  color: var(--white) !important;
}
.stApp {
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 90% 60% at 50% -10%, rgba(176,38,255,0.18) 0%, transparent 65%),
    radial-gradient(ellipse 50% 40% at 90% 90%, rgba(176,38,255,0.07) 0%, transparent 55%),
    radial-gradient(ellipse 40% 30% at 10% 70%, rgba(176,38,255,0.05) 0%, transparent 50%) !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
  padding: 0 2.5rem 4rem !important;
  max-width: 1380px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--neon); border-radius: 3px; }

/* ════════════════════════════════════════════
   NAVBAR
════════════════════════════════════════════ */
.navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.4rem 0 1.2rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0;
  position: sticky;
  top: 0;
  background: rgba(5,5,13,0.92);
  backdrop-filter: blur(18px);
  z-index: 999;
}
.nav-logo {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.nav-logo-icon {
  width: 40px; height: 40px;
  background: linear-gradient(135deg, var(--neon), #7b00d4);
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.2rem;
  box-shadow: 0 0 20px var(--neon-glow);
}
.nav-logo-text {
  font-size: 1.4rem; font-weight: 800;
  letter-spacing: -0.5px; color: var(--white);
}
.nav-logo-text span { color: var(--neon-soft); }
.nav-links {
  display: flex; gap: 2rem;
  align-items: center;
}
.nav-link {
  font-size: 0.9rem; font-weight: 500;
  color: var(--white50); text-decoration: none;
  transition: color 0.2s;
  letter-spacing: 0.02em;
}
.nav-link:hover { color: var(--white); }
.nav-badge {
  background: var(--neon-dim);
  border: 1px solid var(--border);
  color: var(--neon-soft);
  font-size: 0.72rem; font-weight: 700;
  padding: 0.3rem 0.9rem;
  border-radius: 50px;
  letter-spacing: 0.06em;
}

/* ════════════════════════════════════════════
   HERO SECTION
════════════════════════════════════════════ */
.hero {
  text-align: center;
  padding: 5rem 2rem 4rem;
  position: relative;
}
.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: var(--neon-dim);
  border: 1px solid var(--border);
  border-radius: 50px;
  padding: 0.35rem 1rem;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--neon-soft);
  margin-bottom: 1.5rem;
}
.hero-eyebrow::before {
  content: '';
  width: 6px; height: 6px;
  background: var(--neon);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--neon);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
.hero-title {
  font-size: 4.2rem;
  font-weight: 900;
  line-height: 1.05;
  letter-spacing: -2px;
  color: var(--white);
  margin-bottom: 1.2rem;
}
.hero-title .neon-word {
  color: transparent;
  background: linear-gradient(135deg, var(--neon) 0%, #e040fb 50%, var(--neon-soft) 100%);
  -webkit-background-clip: text;
  background-clip: text;
  text-shadow: none;
  filter: drop-shadow(0 0 30px rgba(176,38,255,0.6));
}
.hero-sub {
  font-size: 1.15rem;
  font-weight: 400;
  color: var(--white50);
  max-width: 560px;
  margin: 0 auto 2rem;
  line-height: 1.7;
}
.hero-pills {
  display: flex;
  justify-content: center;
  gap: 0.7rem;
  flex-wrap: wrap;
  margin-bottom: 3rem;
}
.hero-pill {
  background: var(--white04);
  border: 1px solid var(--white08);
  border-radius: 50px;
  padding: 0.4rem 1rem;
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--white50);
}
.hero-pill span { color: var(--neon-soft); margin-right: 0.3rem; }

/* ════════════════════════════════════════════
   HOW IT WORKS STRIP
════════════════════════════════════════════ */
.steps-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--border);
  border-radius: 20px;
  overflow: hidden;
  margin-bottom: 5rem;
  border: 1px solid var(--border);
}
.step {
  background: var(--bg2);
  padding: 1.8rem 1.5rem;
  text-align: center;
  transition: background 0.3s;
}
.step:hover { background: var(--bg3); }
.step-num {
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 700;
  color: var(--neon);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.step-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
.step-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--white);
  margin-bottom: 0.3rem;
}
.step-desc {
  font-size: 0.8rem;
  color: var(--white50);
  line-height: 1.5;
}

/* ════════════════════════════════════════════
   SECTION HEADER
════════════════════════════════════════════ */
.sec-header {
  margin-bottom: 2rem;
}
.sec-label {
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--neon-soft);
  margin-bottom: 0.4rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.sec-label::before {
  content: '';
  display: inline-block;
  width: 20px; height: 2px;
  background: var(--neon);
  border-radius: 2px;
  box-shadow: 0 0 8px var(--neon);
}
.sec-title {
  font-size: 1.8rem;
  font-weight: 800;
  color: var(--white);
  letter-spacing: -0.5px;
  margin-bottom: 0.4rem;
}
.sec-sub {
  font-size: 0.95rem;
  color: var(--white50);
  font-weight: 400;
}

/* ════════════════════════════════════════════
   INPUT CARDS
════════════════════════════════════════════ */
.input-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 1.8rem;
  margin-bottom: 1rem;
  transition: border-color 0.3s;
  position: relative;
  overflow: hidden;
}
.input-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--neon), transparent);
  opacity: 0.6;
}
.input-card:hover { border-color: var(--border-hi); }
.input-card-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1.2rem;
}
.input-card-icon {
  width: 38px; height: 38px;
  background: var(--neon-dim);
  border: 1px solid var(--border);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem;
}
.input-card-title {
  font-size: 1rem;
  font-weight: 700;
  color: var(--white);
}
.input-card-sub {
  font-size: 0.78rem;
  color: var(--white50);
  margin-top: 0.1rem;
}

/* Streamlit file uploader override */
[data-testid="stFileUploadDropzone"] {
  background: rgba(176,38,255,0.04) !important;
  border: 1.5px dashed rgba(176,38,255,0.3) !important;
  border-radius: 14px !important;
  transition: all 0.3s !important;
  min-height: 110px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
  border-color: rgba(176,38,255,0.7) !important;
  background: rgba(176,38,255,0.08) !important;
  box-shadow: 0 0 30px rgba(176,38,255,0.1) !important;
}
[data-testid="stFileUploadDropzone"] p {
  color: var(--white50) !important;
  font-size: 0.9rem !important;
}

/* Streamlit textarea override */
.stTextArea textarea {
  background: rgba(176,38,255,0.04) !important;
  border: 1.5px solid rgba(176,38,255,0.2) !important;
  border-radius: 14px !important;
  color: var(--white) !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.95rem !important;
  line-height: 1.65 !important;
  transition: all 0.3s !important;
  caret-color: var(--neon) !important;
}
.stTextArea textarea:focus {
  border-color: var(--neon) !important;
  box-shadow: 0 0 0 3px rgba(176,38,255,0.15), 0 0 20px rgba(176,38,255,0.1) !important;
}
.stTextArea textarea::placeholder {
  color: rgba(255,255,255,0.25) !important;
}
.stTextArea label { display: none !important; }

/* ════════════════════════════════════════════
   ANALYZE BUTTON
════════════════════════════════════════════ */
.stButton > button {
  background: linear-gradient(135deg, #b026ff 0%, #7b00d4 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 16px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 800 !important;
  font-size: 1.05rem !important;
  letter-spacing: 0.04em !important;
  padding: 1rem 2.5rem !important;
  width: 100% !important;
  transition: all 0.3s !important;
  box-shadow: 0 0 30px rgba(176,38,255,0.4), 0 4px 20px rgba(0,0,0,0.4) !important;
  position: relative !important;
}
.stButton > button:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 0 50px rgba(176,38,255,0.6), 0 8px 30px rgba(0,0,0,0.5) !important;
}
.stButton > button:active {
  transform: translateY(0) !important;
}

/* ════════════════════════════════════════════
   RESULTS SECTION DIVIDER
════════════════════════════════════════════ */
.results-divider {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  margin: 3rem 0 2.5rem;
}
.results-divider-line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}
.results-divider-text {
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--neon-soft);
  white-space: nowrap;
}

/* ════════════════════════════════════════════
   KPI CARDS
════════════════════════════════════════════ */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1rem;
  margin-bottom: 2.5rem;
}
.kpi-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.4rem 1.2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: all 0.3s;
}
.kpi-card::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 18px;
  background: radial-gradient(ellipse at 50% 0%, rgba(176,38,255,0.1), transparent 70%);
  pointer-events: none;
}
.kpi-card:hover {
  border-color: var(--border-hi);
  transform: translateY(-3px);
  box-shadow: 0 8px 40px rgba(176,38,255,0.12);
}
.kpi-icon {
  font-size: 1.4rem;
  margin-bottom: 0.5rem;
  display: block;
}
.kpi-val {
  font-family: var(--mono);
  font-size: 2rem;
  font-weight: 700;
  color: var(--neon-soft);
  line-height: 1;
  margin-bottom: 0.35rem;
}
.kpi-label {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--white50);
  letter-spacing: 0.05em;
}
.kpi-sub {
  font-size: 0.7rem;
  color: var(--neon-soft);
  margin-top: 0.25rem;
  opacity: 0.7;
}

/* ════════════════════════════════════════════
   TABS
════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 0.3rem !important;
  gap: 0.2rem !important;
  margin-bottom: 2rem !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.04em !important;
  color: var(--white50) !important;
  background: transparent !important;
  border-radius: 10px !important;
  padding: 0.65rem 1.4rem !important;
  transition: all 0.2s !important;
  border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--white) !important;
  background: var(--white04) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--white) !important;
  background: linear-gradient(135deg, var(--neon), #7b00d4) !important;
  box-shadow: 0 0 20px rgba(176,38,255,0.4) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 0 !important; }

/* ════════════════════════════════════════════
   CANDIDATE CARDS
════════════════════════════════════════════ */
.c-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 2rem 2.2rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
  transition: all 0.3s;
}
.c-card:hover {
  border-color: var(--border-hi);
  box-shadow: 0 0 50px rgba(176,38,255,0.08);
  transform: translateY(-2px);
}
.c-card-glow {
  background: var(--bg2);
  border: 1px solid var(--neon);
  border-radius: 22px;
  padding: 2rem 2.2rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 0 40px rgba(176,38,255,0.18), inset 0 0 40px rgba(176,38,255,0.03);
}
.c-card-stripe {
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--neon), #e040fb, var(--neon-soft));
  box-shadow: 0 0 15px rgba(176,38,255,0.8);
}
.c-rank-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  background: var(--neon-dim);
  border: 1px solid var(--border-hi);
  border-radius: 50px;
  padding: 0.25rem 0.8rem;
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--neon-soft);
  margin-bottom: 0.6rem;
}
.c-rank-badge.gold {
  background: rgba(255,190,11,0.1);
  border-color: rgba(255,190,11,0.4);
  color: #ffbe0b;
}
.c-name {
  font-size: 1.4rem;
  font-weight: 800;
  color: var(--white);
  letter-spacing: -0.3px;
  margin-bottom: 0.2rem;
}
.c-file {
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--white20);
}
.c-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  margin-top: 0.7rem;
}
.c-meta-item {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.8rem;
  color: var(--white50);
}
.c-meta-item a {
  color: var(--neon-soft);
  text-decoration: none;
}
.c-meta-item a:hover { text-decoration: underline; }

/* Score badge */
.score-ring {
  text-align: center;
}
.score-big {
  font-family: var(--mono);
  font-size: 2.4rem;
  font-weight: 700;
  line-height: 1;
}
.score-big.hi  { color: var(--green);  text-shadow: 0 0 20px rgba(0,245,160,0.5); }
.score-big.mid { color: var(--amber);  text-shadow: 0 0 20px rgba(255,190,11,0.5); }
.score-big.lo  { color: var(--red);    text-shadow: 0 0 20px rgba(255,77,109,0.5); }
.score-label {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--white50);
  margin-top: 0.2rem;
}

/* Score bars */
.score-bars { margin: 1.2rem 0; }
.sbar-row {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin-bottom: 0.7rem;
}
.sbar-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--white50);
  width: 100px;
  flex-shrink: 0;
}
.sbar-track {
  flex: 1;
  height: 6px;
  background: var(--white08);
  border-radius: 4px;
  overflow: hidden;
}
.sbar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.8s ease;
}
.sbar-fill.neon  { background: linear-gradient(90deg, var(--neon), #e040fb); box-shadow: 0 0 10px rgba(176,38,255,0.5); }
.sbar-fill.green { background: linear-gradient(90deg, #00c9ff, var(--green)); box-shadow: 0 0 10px rgba(0,245,160,0.4); }
.sbar-fill.amber { background: linear-gradient(90deg, #ff9a00, var(--amber)); box-shadow: 0 0 10px rgba(255,190,11,0.4); }
.sbar-val {
  font-family: var(--mono);
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--white80);
  width: 36px;
  text-align: right;
  flex-shrink: 0;
}

/* Divider */
.c-divider {
  height: 1px;
  background: var(--border);
  margin: 1.2rem 0;
}

/* Section inside card */
.c-section-label {
  font-size: 0.65rem;
  font-weight: 800;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--neon-soft);
  margin-bottom: 0.6rem;
  opacity: 0.8;
}

/* Skill pills */
.pill {
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.28rem 0.75rem;
  border-radius: 50px;
  margin: 0.18rem;
  letter-spacing: 0.02em;
  transition: all 0.2s;
}
.pill:hover { transform: translateY(-1px); }
.p-found {
  background: rgba(176,38,255,0.1);
  border: 1px solid rgba(176,38,255,0.3);
  color: var(--neon-soft);
}
.p-missing {
  background: rgba(255,77,109,0.08);
  border: 1px solid rgba(255,77,109,0.25);
  color: #ff8fa3;
}

/* Suggestion item */
.tip-item {
  display: flex;
  gap: 0.8rem;
  background: rgba(255,190,11,0.04);
  border: 1px solid rgba(255,190,11,0.12);
  border-left: 3px solid var(--amber);
  border-radius: 0 12px 12px 0;
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
  font-size: 0.88rem;
  color: var(--white80);
  line-height: 1.55;
  transition: background 0.2s;
}
.tip-item:hover { background: rgba(255,190,11,0.07); }
.tip-arrow {
  color: var(--amber);
  font-weight: 800;
  flex-shrink: 0;
  margin-top: 0.05rem;
}

/* ════════════════════════════════════════════
   PROGRESS BAR OVERRIDE
════════════════════════════════════════════ */
.stProgress > div {
  background: var(--white08) !important;
  border-radius: 4px !important;
  height: 4px !important;
}
.stProgress > div > div {
  background: linear-gradient(90deg, var(--neon), #e040fb) !important;
  border-radius: 4px !important;
  box-shadow: 0 0 10px rgba(176,38,255,0.5) !important;
}

/* ════════════════════════════════════════════
   COMPARISON TABLE
════════════════════════════════════════════ */
.cmp-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}
.cmp-table th {
  padding: 0.8rem 1.2rem;
  text-align: left;
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--neon-soft);
  border-bottom: 1px solid var(--border);
}
.cmp-table td {
  padding: 0.85rem 1.2rem;
  color: var(--white80);
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 0.88rem;
  font-weight: 500;
}
.cmp-table tr:hover td { background: var(--neon-dim); }
.cmp-table .win { color: var(--green); font-weight: 700; }
.cmp-table .lose { color: var(--white20); }

/* ════════════════════════════════════════════
   DOWNLOAD BUTTONS
════════════════════════════════════════════ */
.stDownloadButton > button {
  background: transparent !important;
  color: var(--neon-soft) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 12px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  padding: 0.65rem 1.2rem !important;
  width: 100% !important;
  transition: all 0.25s !important;
}
.stDownloadButton > button:hover {
  background: var(--neon-dim) !important;
  border-color: var(--neon) !important;
  box-shadow: 0 0 20px rgba(176,38,255,0.2) !important;
  transform: translateY(-2px) !important;
}

/* ════════════════════════════════════════════
   DATAFRAME
════════════════════════════════════════════ */
.stDataFrame {
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
}

/* ════════════════════════════════════════════
   SELECTBOX
════════════════════════════════════════════ */
.stSelectbox > div > div {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--white) !important;
}

/* ════════════════════════════════════════════
   FOOTER
════════════════════════════════════════════ */
.site-footer {
  text-align: center;
  padding: 3rem 0 2rem;
  border-top: 1px solid var(--border);
  margin-top: 4rem;
}
.site-footer-logo {
  font-size: 1.1rem;
  font-weight: 800;
  color: var(--neon-soft);
  margin-bottom: 0.5rem;
}
.site-footer-sub {
  font-size: 0.8rem;
  color: var(--white20);
}

/* ════════════════════════════════════════════
   ANIMATIONS
════════════════════════════════════════════ */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
.anim-up { animation: fadeUp 0.5s ease forwards; }
.anim-in { animation: fadeIn 0.4s ease forwards; }

/* ════════════════════════════════════════════
   SPINNER OVERRIDE
════════════════════════════════════════════ */
.stSpinner > div { border-top-color: var(--neon) !important; }

/* Info/warning/error */
.stAlert { border-radius: 12px !important; border-left-width: 4px !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="nav-logo">
    <div class="nav-logo-icon">⚡</div>
    <div>
      <div class="nav-logo-text">Resume<span>AI</span></div>
    </div>
    <div class="nav-badge">BETA v2.0</div>
  </div>
  <div class="nav-links">
    <span class="nav-link">For HR Teams</span>
    <span class="nav-link">For Job Seekers</span>
    <span class="nav-link">Features</span>
    <span class="nav-link">About</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero anim-up">
  <div class="hero-eyebrow">⚡ Powered by BERT AI + ATS Engine</div>
  <div class="hero-title">
    Screen Resumes<br>
    <span class="neon-word">10x Faster</span> with AI
  </div>
  <div class="hero-sub">
    Upload resumes, paste a job description, and get instant AI-powered rankings,
    skill gap analysis, ATS scores, and actionable feedback — in seconds.
  </div>
  <div class="hero-pills">
    <div class="hero-pill"><span>🧠</span> BERT Semantic Matching</div>
    <div class="hero-pill"><span>📊</span> ATS Score Analysis</div>
    <div class="hero-pill"><span>🔍</span> Skill Gap Detection</div>
    <div class="hero-pill"><span>📈</span> Candidate Ranking</div>
    <div class="hero-pill"><span>💡</span> AI Suggestions</div>
    <div class="hero-pill"><span>📥</span> Export Reports</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="steps-strip">
  <div class="step">
    <div class="step-num">Step 01</div>
    <div class="step-icon">📄</div>
    <div class="step-title">Upload Resumes</div>
    <div class="step-desc">Drag & drop PDF or DOCX files. Multiple resumes supported.</div>
  </div>
  <div class="step">
    <div class="step-num">Step 02</div>
    <div class="step-icon">📋</div>
    <div class="step-title">Paste Job Description</div>
    <div class="step-desc">Copy-paste the full JD. More detail = more accurate results.</div>
  </div>
  <div class="step">
    <div class="step-num">Step 03</div>
    <div class="step-icon">⚡</div>
    <div class="step-title">AI Analyzes</div>
    <div class="step-desc">BERT embeddings + ATS scoring + skill extraction runs instantly.</div>
  </div>
  <div class="step">
    <div class="step-num">Step 04</div>
    <div class="step-icon">🏆</div>
    <div class="step-title">Get Ranked Results</div>
    <div class="step-desc">See scores, gaps, suggestions, charts & download reports.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-header">
  <div class="sec-label">Get Started</div>
  <div class="sec-title">Upload & Analyze</div>
  <div class="sec-sub">Upload resumes and paste the job description below to begin AI screening.</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="input-card">
      <div class="input-card-header">
        <div class="input-card-icon">📄</div>
        <div>
          <div class="input-card-title">Resume Files</div>
          <div class="input-card-sub">PDF or DOCX · Multiple files supported</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if files:
        count = len(files) if isinstance(files, list) else 1
        st.markdown(
            f"<div style='margin-top:0.6rem;padding:0.6rem 1rem;"
            f"background:rgba(0,245,160,0.06);border:1px solid rgba(0,245,160,0.2);"
            f"border-radius:10px;font-size:0.85rem;color:#00f5a0;font-weight:600'>"
            f"✓ {count} resume{'s' if count>1 else ''} ready for analysis</div>",
            unsafe_allow_html=True
        )

with col_right:
    st.markdown("""
    <div class="input-card">
      <div class="input-card-header">
        <div class="input-card-icon">💼</div>
        <div>
          <div class="input-card-title">Job Description</div>
          <div class="input-card-sub">Paste the full JD — more detail = better accuracy</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    jd = st.text_area(
        "JD",
        height=180,
        placeholder="We are looking for a Python Developer with experience in FastAPI, PostgreSQL, Docker, AWS...\n\nThe more detail you add, the more accurate the AI matching will be.",
        label_visibility="collapsed",
    )
    if jd:
        wc = len(jd.split())
        col = "#00f5a0" if wc > 60 else "#ffbe0b" if wc > 20 else "#ff4d6d"
        icon = "✓" if wc > 60 else "⚠"
        msg = "Great detail — high accuracy expected" if wc > 60 else "Add more detail for better results" if wc > 20 else "Too short — add more description"
        st.markdown(
            f"<div style='margin-top:0.5rem;font-size:0.78rem;color:{col};font-weight:600'>"
            f"{icon} {wc} words — {msg}</div>",
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

_, btn_col, _ = st.columns([1.5, 1, 1.5])
with btn_col:
    analyze = st.button("⚡  Analyze Resumes Now")

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
if analyze:
    file_list = files if isinstance(files, list) else ([files] if files else [])

    if not file_list:
        st.warning("⚠️  Please upload at least one resume file.")
        st.stop()
    if not jd or len(jd.split()) < 10:
        st.warning("⚠️  Please enter a job description with at least 10 words.")
        st.stop()

    results, errors = [], []
    jd_skills = extract_skills(jd)
    kw_density_jd = compute_keyword_density(jd)

    progress_bar = st.progress(0)
    status = st.empty()

    for idx, f in enumerate(file_list):
        status.markdown(
            f"<div style='text-align:center;color:var(--white50);font-size:0.85rem;margin:0.5rem 0'>"
            f"🔬 Analyzing <strong style='color:white'>{f.name}</strong> "
            f"({idx+1}/{len(file_list)})...</div>",
            unsafe_allow_html=True
        )
        progress_bar.progress((idx + 1) / len(file_list))

        try:
            raw  = f.read()
            text = parse_resume(raw, f.name)
            if not text or len(text.strip()) < 40:
                errors.append((f.name, "Could not extract text — ensure PDF is not scanned/image-based"))
                continue

            bert_score   = compute_similarity(text, jd)
            skills_found = extract_skills(text)
            missing      = get_missing_skills(skills_found, jd_skills)
            ats          = compute_ats_score(text, jd)
            skill_pct    = round((len(skills_found)/max(len(jd_skills),1))*100, 1) if jd_skills else 0.0
            composite    = round(bert_score*100*0.5 + skill_pct*0.3 + ats*0.2, 1)
            tips         = generate_suggestions(bert_score, skills_found, missing, ats)
            exp_years    = extract_experience_years(text)
            github       = extract_github(text)
            email        = extract_email(text)
            phone        = extract_phone(text)
            name         = f.name.rsplit(".",1)[0].replace("_"," ").replace("-"," ").title()

            results.append({
                "name": name, "filename": f.name,
                "score": composite,
                "bert": round(bert_score*100, 1),
                "ats": round(ats, 1),
                "skill_pct": skill_pct,
                "skills": skills_found,
                "missing": missing,
                "suggestions": tips,
                "exp_years": exp_years,
                "github": github,
                "email": email,
                "phone": phone,
                "text": text,
            })
        except Exception as e:
            errors.append((f.name, str(e)[:80]))

    progress_bar.empty()
    status.empty()

    if errors:
        for fname, reason in errors:
            st.error(f"⚠️  **{fname}** — {reason}")

    if not results:
        st.error("No resumes could be analyzed. Please check the file format.")
        st.stop()

    results.sort(key=lambda x: x["score"], reverse=True)

    # ── RESULTS DIVIDER ──────────────────────────────────────────────────
    st.markdown("""
    <div class="results-divider anim-in">
      <div class="results-divider-line"></div>
      <div class="results-divider-text">⚡ AI Analysis Complete</div>
      <div class="results-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI CARDS ────────────────────────────────────────────────────────
    avg   = round(sum(r["score"] for r in results)/len(results), 1)
    qual  = sum(1 for r in results if r["score"] >= 60)
    top   = results[0]
    avgats= round(sum(r["ats"] for r in results)/len(results), 1)

    st.markdown(f"""
    <div class="kpi-grid anim-up">
      <div class="kpi-card">
        <span class="kpi-icon">📋</span>
        <div class="kpi-val">{len(results)}</div>
        <div class="kpi-label">Resumes Screened</div>
        <div class="kpi-sub">{len(errors)} skipped</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-icon">🎯</span>
        <div class="kpi-val">{avg}%</div>
        <div class="kpi-label">Average Match</div>
        <div class="kpi-sub">Composite score</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-icon">🏆</span>
        <div class="kpi-val">{top['score']}%</div>
        <div class="kpi-label">Top Score</div>
        <div class="kpi-sub">{top['name']}</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-icon">✅</span>
        <div class="kpi-val">{qual}</div>
        <div class="kpi-label">Qualified ≥ 60%</div>
        <div class="kpi-sub">{round(qual/len(results)*100)}% pass rate</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-icon">🤖</span>
        <div class="kpi-val">{avgats}%</div>
        <div class="kpi-label">Avg ATS Score</div>
        <div class="kpi-sub">Keyword density</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆  Candidate Rankings",
        "📊  Analytics & Charts",
        "⚖️  Side-by-Side Compare",
        "📥  Export Reports",
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — RANKINGS
    # ════════════════════════════════════════════════════════════════════
    with tab1:
        medals = ["🥇", "🥈", "🥉"]
        rank_labels = ["TOP CANDIDATE", "2ND PLACE", "3RD PLACE"]

        for i, r in enumerate(results):
            s      = r["score"]
            s_cls  = "hi" if s >= 70 else "mid" if s >= 45 else "lo"
            s_word = "Excellent" if s >= 70 else "Good" if s >= 45 else "Low"
            medal  = medals[i] if i < 3 else f"#{i+1}"
            is_top = i == 0

            # Skill pills
            found_pills = "".join(f'<span class="pill p-found">{x}</span>' for x in r["skills"]) \
                or "<span style='color:var(--white20);font-size:0.82rem'>No skills detected</span>"
            miss_pills  = "".join(f'<span class="pill p-missing">{x}</span>' for x in r["missing"]) \
                or "<span style='color:var(--green);font-size:0.82rem'>✓ All required skills matched!</span>"
            tips_html   = "".join(
                f'<div class="tip-item"><div class="tip-arrow">→</div><div>{t}</div></div>'
                for t in r["suggestions"]
            )

            # Meta chips
            meta = []
            if r["exp_years"]: meta.append(f'<div class="c-meta-item">🕐 <strong>{r["exp_years"]} years</strong> experience</div>')
            if r["email"]:     meta.append(f'<div class="c-meta-item">✉️ {r["email"]}</div>')
            if r["phone"]:     meta.append(f'<div class="c-meta-item">📞 {r["phone"]}</div>')
            if r["github"]:    meta.append(f'<div class="c-meta-item">🐙 <a href="https://{r["github"]}" target="_blank">{r["github"]}</a></div>')
            meta_html = "".join(meta)

            card_cls = "c-card-glow" if is_top else "c-card"
            rank_badge_cls = "gold" if is_top else ""
            rank_txt = rank_labels[i] if i < 3 else f"RANK {i+1}"

            st.markdown(f"""
            <div class="{card_cls} anim-up">
              <div class="c-card-stripe"></div>

              <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem">
                <div style="flex:1">
                  <div class="c-rank-badge {rank_badge_cls}">{medal} {rank_txt}</div>
                  <div class="c-name">{r['name']}</div>
                  <div class="c-file">{r['filename']}</div>
                  <div class="c-meta">{meta_html}</div>
                </div>
                <div class="score-ring">
                  <div class="score-big {s_cls}">{s}%</div>
                  <div class="score-label">{s_word} Match</div>
                </div>
              </div>

              <div class="score-bars">
                <div class="sbar-row">
                  <div class="sbar-label">🧠 BERT AI</div>
                  <div class="sbar-track"><div class="sbar-fill neon" style="width:{r['bert']}%"></div></div>
                  <div class="sbar-val">{r['bert']}%</div>
                </div>
                <div class="sbar-row">
                  <div class="sbar-label">🛠 Skills</div>
                  <div class="sbar-track"><div class="sbar-fill green" style="width:{r['skill_pct']}%"></div></div>
                  <div class="sbar-val">{r['skill_pct']}%</div>
                </div>
                <div class="sbar-row">
                  <div class="sbar-label">📋 ATS Score</div>
                  <div class="sbar-track"><div class="sbar-fill amber" style="width:{r['ats']}%"></div></div>
                  <div class="sbar-val">{r['ats']}%</div>
                </div>
              </div>

              <div class="c-divider"></div>

              <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.2rem">
                <div>
                  <div class="c-section-label">✅ Matched Skills ({len(r['skills'])} found)</div>
                  <div>{found_pills}</div>
                </div>
                <div>
                  <div class="c-section-label">❌ Missing Skills ({len(r['missing'])} gaps)</div>
                  <div>{miss_pills}</div>
                </div>
              </div>

              <div class="c-divider"></div>

              <div>
                <div class="c-section-label">💡 AI-Powered Suggestions</div>
                {tips_html}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(min(s, 100)) / 100)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — ANALYTICS
    # ════════════════════════════════════════════════════════════════════
    with tab2:
        CHART_LAYOUT = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.5)", family="Outfit"),
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.markdown('<div class="sec-label" style="margin-bottom:1.2rem">Score Overview</div>',
                    unsafe_allow_html=True)
        ca, cb = st.columns(2, gap="medium")

        with ca:
            df = pd.DataFrame(results)[["name","score"]].sort_values("score")
            colors_bar = ["#00f5a0" if s>=70 else "#ffbe0b" if s>=45 else "#ff4d6d"
                          for s in df["score"]]
            fig1 = go.Figure(go.Bar(
                x=df["score"], y=df["name"], orientation="h",
                marker=dict(
                    color=colors_bar,
                    line=dict(width=0),
                ),
                text=[f"{s}%" for s in df["score"]],
                textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.7)", size=12, family="Outfit"),
            ))
            fig1.update_layout(
                **CHART_LAYOUT,
                title=dict(text="Composite Match Score", font=dict(color="rgba(255,255,255,0.6)", size=13)),
                xaxis=dict(showgrid=True, gridcolor="rgba(176,38,255,0.08)",
                           ticksuffix="%", range=[0, 118],
                           tickfont=dict(size=11, color="rgba(255,255,255,0.3)")),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color="rgba(255,255,255,0.5)")),
                height=max(280, len(results)*55),
            )
            st.plotly_chart(fig1, use_container_width=True)

        with cb:
            fig2 = go.Figure()
            names = [r["name"] for r in results]
            fig2.add_trace(go.Bar(name="BERT AI", x=names,
                                  y=[r["bert"] for r in results],
                                  marker_color="rgba(176,38,255,0.8)"))
            fig2.add_trace(go.Bar(name="Skills", x=names,
                                  y=[r["skill_pct"] for r in results],
                                  marker_color="rgba(0,245,160,0.7)"))
            fig2.add_trace(go.Bar(name="ATS", x=names,
                                  y=[r["ats"] for r in results],
                                  marker_color="rgba(255,190,11,0.7)"))
            fig2.update_layout(
                **CHART_LAYOUT,
                barmode="group",
                title=dict(text="Score Breakdown", font=dict(color="rgba(255,255,255,0.6)", size=13)),
                legend=dict(orientation="h", x=0, y=1.15,
                            font=dict(size=11, color="rgba(255,255,255,0.5)"),
                            bgcolor="rgba(0,0,0,0)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(176,38,255,0.08)",
                           ticksuffix="%", range=[0, 110],
                           tickfont=dict(size=11, color="rgba(255,255,255,0.3)")),
                xaxis=dict(showgrid=False, tickfont=dict(size=10, color="rgba(255,255,255,0.4)")),
                height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Skill frequency
        st.markdown('<div class="sec-label" style="margin:1.5rem 0 1rem">Top Skills Across All Resumes</div>',
                    unsafe_allow_html=True)
        all_skills: dict[str, int] = {}
        for r in results:
            for s in r["skills"]:
                all_skills[s] = all_skills.get(s, 0) + 1
        if all_skills:
            df_sk = pd.DataFrame(
                sorted(all_skills.items(), key=lambda x: -x[1])[:18],
                columns=["Skill", "Count"]
            )
            fig3 = px.bar(df_sk, x="Skill", y="Count",
                          color="Count",
                          color_continuous_scale=["#3b0066", "#b026ff", "#e040fb", "#00f5a0"])
            fig3.update_layout(
                **CHART_LAYOUT,
                coloraxis_showscale=False,
                yaxis=dict(showgrid=True, gridcolor="rgba(176,38,255,0.08)",
                           tickfont=dict(size=11, color="rgba(255,255,255,0.3)")),
                xaxis=dict(showgrid=False, tickfont=dict(size=10, color="rgba(255,255,255,0.5)")),
                height=300,
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Keyword density
        st.markdown('<div class="sec-label" style="margin:1.5rem 0 1rem">JD Keyword Coverage in Resumes</div>',
                    unsafe_allow_html=True)
        jd_kws = sorted(kw_density_jd.items(), key=lambda x: -x[1])[:12]
        if jd_kws:
            kc1, kc2 = st.columns(2, gap="large")
            for idx2, (kw, _) in enumerate(jd_kws):
                found_in = sum(1 for r in results if kw.lower() in r["text"].lower())
                pct2     = round(found_in / len(results) * 100)
                col2     = kc1 if idx2 % 2 == 0 else kc2
                bar_color = "#00f5a0" if pct2 >= 70 else "#ffbe0b" if pct2 >= 40 else "#ff4d6d"
                with col2:
                    st.markdown(f"""
                    <div style="margin-bottom:0.8rem">
                      <div style="display:flex;justify-content:space-between;
                                  font-size:0.8rem;margin-bottom:0.3rem">
                        <span style="color:rgba(255,255,255,0.8);font-weight:600">{kw}</span>
                        <span style="color:rgba(255,255,255,0.3)">{found_in}/{len(results)} ({pct2}%)</span>
                      </div>
                      <div style="height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden">
                        <div style="width:{pct2}%;height:100%;background:{bar_color};
                                    border-radius:3px;box-shadow:0 0 8px {bar_color}55"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Scatter
        if len(results) > 1:
            st.markdown('<div class="sec-label" style="margin:2rem 0 1rem">BERT vs ATS — Candidate Positioning</div>',
                        unsafe_allow_html=True)
            df_sc = pd.DataFrame({
                "Candidate": [r["name"] for r in results],
                "BERT (%)":  [r["bert"] for r in results],
                "ATS (%)":   [r["ats"]  for r in results],
                "Score":     [r["score"] for r in results],
            })
            fig4 = px.scatter(df_sc, x="BERT (%)", y="ATS (%)",
                              size="Score", color="Score", text="Candidate",
                              color_continuous_scale=["#ff4d6d", "#ffbe0b", "#b026ff", "#00f5a0"],
                              size_max=45)
            fig4.update_traces(
                textposition="top center",
                textfont=dict(size=9, color="rgba(255,255,255,0.6)", family="Outfit"),
            )
            fig4.update_layout(
                **CHART_LAYOUT,
                xaxis=dict(showgrid=True, gridcolor="rgba(176,38,255,0.08)",
                           ticksuffix="%", title="BERT Semantic Score",
                           tickfont=dict(size=11, color="rgba(255,255,255,0.3)")),
                yaxis=dict(showgrid=True, gridcolor="rgba(176,38,255,0.08)",
                           ticksuffix="%", title="ATS Keyword Score",
                           tickfont=dict(size=11, color="rgba(255,255,255,0.3)")),
                coloraxis_showscale=False,
                height=400,
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — COMPARE
    # ════════════════════════════════════════════════════════════════════
    with tab3:
        if len(results) < 2:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem">
              <div style="font-size:3rem;margin-bottom:1rem;opacity:0.3">⚖️</div>
              <div style="font-size:1.1rem;font-weight:700;color:rgba(255,255,255,0.4)">
                Upload 2 or more resumes to use this feature
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="sec-label" style="margin-bottom:1rem">Select Two Candidates</div>',
                        unsafe_allow_html=True)
            cc1, cc2 = st.columns(2, gap="large")
            names_list = [r["name"] for r in results]
            with cc1:
                a_name = st.selectbox("Candidate A", names_list, index=0, key="ca")
            with cc2:
                b_name = st.selectbox("Candidate B", names_list,
                                      index=min(1, len(names_list)-1), key="cb")

            ra = next(r for r in results if r["name"] == a_name)
            rb = next(r for r in results if r["name"] == b_name)

            # Radar
            st.markdown('<div class="sec-label" style="margin:1.5rem 0 1rem">Radar Comparison</div>',
                        unsafe_allow_html=True)
            cats = ["Composite", "BERT AI", "ATS Score", "Skill Cover", "Completeness"]
            va   = [ra["score"], ra["bert"], ra["ats"], ra["skill_pct"],
                    max(0, 100 - len(ra["missing"])*8)]
            vb   = [rb["score"], rb["bert"], rb["ats"], rb["skill_pct"],
                    max(0, 100 - len(rb["missing"])*8)]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=va+[va[0]], theta=cats+[cats[0]],
                fill="toself", name=a_name,
                line=dict(color="#b026ff", width=2.5),
                fillcolor="rgba(176,38,255,0.15)",
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=vb+[vb[0]], theta=cats+[cats[0]],
                fill="toself", name=b_name,
                line=dict(color="#00f5a0", width=2.5),
                fillcolor="rgba(0,245,160,0.1)",
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0,100],
                                   tickfont=dict(size=8, color="rgba(255,255,255,0.2)"),
                                   gridcolor="rgba(176,38,255,0.12)"),
                    angularaxis=dict(tickfont=dict(size=11, color="rgba(255,255,255,0.5)"),
                                     gridcolor="rgba(176,38,255,0.1)"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit", color="rgba(255,255,255,0.5)"),
                legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)",
                            orientation="h", x=0.3, y=-0.05),
                margin=dict(l=20, r=20, t=20, b=40),
                height=400,
            )
            st.plotly_chart(fig_r, use_container_width=True)

            # Table
            st.markdown('<div class="sec-label" style="margin:1.5rem 0 1rem">Head-to-Head Stats</div>',
                        unsafe_allow_html=True)

            def w(av, bv, hi=True):
                if av == bv:
                    return str(av), str(bv)
                a_wins = (av > bv) == hi
                if a_wins:
                    return f"<span class='win'>{av} ↑</span>", f"<span class='lose'>{bv}</span>"
                return f"<span class='lose'>{av}</span>", f"<span class='win'>{bv} ↑</span>"

            rows = [
                ("Composite Score", *w(ra["score"], rb["score"]), "%"),
                ("BERT Similarity", *w(ra["bert"], rb["bert"]), "%"),
                ("ATS Score",       *w(ra["ats"],  rb["ats"]),  "%"),
                ("Skill Coverage",  *w(ra["skill_pct"], rb["skill_pct"]), "%"),
                ("Missing Skills",  *w(len(ra["missing"]), len(rb["missing"]), hi=False), ""),
                ("Experience",
                    f"{ra['exp_years']} yrs" if ra["exp_years"] else "?",
                    f"{rb['exp_years']} yrs" if rb["exp_years"] else "?",
                    ""),
                ("GitHub",
                    f'<a href="https://{ra["github"]}" style="color:#b026ff">✓</a>' if ra["github"] else "✗",
                    f'<a href="https://{rb["github"]}" style="color:#b026ff">✓</a>' if rb["github"] else "✗",
                    ""),
            ]

            rows_html = "".join(
                f"<tr><td style='color:rgba(255,255,255,0.4);font-size:0.82rem'>{label}</td>"
                f"<td>{av}{unit}</td><td>{bv}{unit}</td></tr>"
                for label, av, bv, unit in rows
            )
            st.markdown(f"""
            <table class="cmp-table">
              <tr>
                <th>Metric</th>
                <th style="color:#b026ff">{a_name}</th>
                <th style="color:#00f5a0">{b_name}</th>
              </tr>
              {rows_html}
            </table>
            """, unsafe_allow_html=True)

            # Skill sets
            st.markdown('<div class="sec-label" style="margin:2rem 0 1rem">Skill Set Analysis</div>',
                        unsafe_allow_html=True)
            sa_set  = set(ra["skills"])
            sb_set  = set(rb["skills"])
            only_a  = sa_set - sb_set
            only_b  = sb_set - sa_set
            shared  = sa_set & sb_set

            sk1, sk2, sk3 = st.columns(3, gap="medium")
            with sk1:
                st.markdown(f'<div class="c-section-label" style="margin-bottom:0.5rem">Only {a_name}</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    "".join(f'<span class="pill p-found">{s}</span>' for s in sorted(only_a))
                    or "<span style='color:var(--white20);font-size:0.8rem'>None</span>",
                    unsafe_allow_html=True
                )
            with sk2:
                st.markdown('<div class="c-section-label" style="margin-bottom:0.5rem">Shared Skills</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    "".join(f'<span class="pill" style="background:rgba(0,245,160,0.08);border:1px solid rgba(0,245,160,0.25);color:#00f5a0">{s}</span>' for s in sorted(shared))
                    or "<span style='color:var(--white20);font-size:0.8rem'>None</span>",
                    unsafe_allow_html=True
                )
            with sk3:
                st.markdown(f'<div class="c-section-label" style="margin-bottom:0.5rem">Only {b_name}</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    "".join(f'<span class="pill p-found">{s}</span>' for s in sorted(only_b))
                    or "<span style='color:var(--white20);font-size:0.8rem'>None</span>",
                    unsafe_allow_html=True
                )

    # ════════════════════════════════════════════════════════════════════
    # TAB 4 — EXPORT
    # ════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("""
        <div class="sec-header" style="margin-bottom:1.5rem">
          <div class="sec-label">Export</div>
          <div class="sec-title">Download Your Report</div>
          <div class="sec-sub">Save results in your preferred format for sharing or record-keeping.</div>
        </div>
        """, unsafe_allow_html=True)

        e1, e2, e3 = st.columns(3, gap="medium")
        with e1:
            st.markdown("""
            <div class="input-card" style="text-align:center;padding:1.5rem 1rem">
              <div style="font-size:2rem;margin-bottom:0.5rem">📋</div>
              <div style="font-weight:700;color:white;margin-bottom:0.3rem">JSON Report</div>
              <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);margin-bottom:1rem">
                Full data with all scores, skills and suggestions
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.download_button("⬇️  Download JSON",
                               data=to_json(results),
                               file_name="resumeai_report.json",
                               mime="application/json",
                               use_container_width=True)
        with e2:
            st.markdown("""
            <div class="input-card" style="text-align:center;padding:1.5rem 1rem">
              <div style="font-size:2rem;margin-bottom:0.5rem">📊</div>
              <div style="font-weight:700;color:white;margin-bottom:0.3rem">CSV Report</div>
              <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);margin-bottom:1rem">
                Open in Excel or Google Sheets
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.download_button("⬇️  Download CSV",
                               data=to_csv(results),
                               file_name="resumeai_report.csv",
                               mime="text/csv",
                               use_container_width=True)
        with e3:
            st.markdown("""
            <div class="input-card" style="text-align:center;padding:1.5rem 1rem">
              <div style="font-size:2rem;margin-bottom:0.5rem">📝</div>
              <div style="font-weight:700;color:white;margin-bottom:0.3rem">Text Summary</div>
              <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);margin-bottom:1rem">
                Formatted report — save as PDF
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.download_button("⬇️  Download TXT",
                               data=to_pdf_report(results),
                               file_name="resumeai_summary.txt",
                               mime="text/plain",
                               use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label" style="margin-bottom:1rem">Results Preview Table</div>',
                    unsafe_allow_html=True)
        df_show = pd.DataFrame([{
            "Rank":           i + 1,
            "Candidate":      r["name"],
            "Score (%)":      r["score"],
            "BERT (%)":       r["bert"],
            "ATS (%)":        r["ats"],
            "Skills Found":   len(r["skills"]),
            "Skills Missing": len(r["missing"]),
            "Experience":     f"{r['exp_years']} yrs" if r["exp_years"] else "?",
            "GitHub":         "✓" if r["github"] else "–",
            "Email":          r["email"] or "–",
        } for i, r in enumerate(results)])

        st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score (%)": st.column_config.ProgressColumn(
                    "Score (%)", min_value=0, max_value=100, format="%d%%"),
                "BERT (%)":  st.column_config.ProgressColumn(
                    "BERT (%)",  min_value=0, max_value=100, format="%d%%"),
                "ATS (%)":   st.column_config.ProgressColumn(
                    "ATS (%)",   min_value=0, max_value=100, format="%d%%"),
            }
        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
  <div class="site-footer-logo">⚡ ResumeAI</div>
  <div class="site-footer-sub">
    Powered by BERT · Built with Streamlit · AI Resume Intelligence Platform
  </div>
</div>
""", unsafe_allow_html=True)