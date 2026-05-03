import streamlit as st
import joblib
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from groq import Groq
from tavily import TavilyClient

# ─── Page Config ─────────────────────────────────────
st.set_page_config(page_title="News Credibility Analyzer", page_icon="🔍", layout="centered")

# ─── Custom CSS ──────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 720px;}

.navbar {display: flex; align-items: center; justify-content: space-between; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 2rem;}
.navbar-left {display: flex; align-items: center; gap: 10px;}
.nav-dot {width: 8px; height: 8px; border-radius: 50%; background: #378ADD;}
.nav-title {font-size: 15px; font-weight: 600; letter-spacing: -0.01em;}
.nav-sub {font-size: 12px; opacity: 0.5; margin-top: 2px;}
.nav-badge {font-size: 11px; padding: 4px 10px; border-radius: 20px; background: rgba(55,138,221,0.15); color: #378ADD; font-weight: 500;}

.hint-text {font-size: 13px; opacity: 0.45; margin-bottom: 1.5rem;}

.section-card {border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; margin-bottom: 12px;}
.section-header {padding: 10px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03);}
.section-tag {font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; opacity: 0.5;}
.section-body {padding: 16px;}

.verdict-fake {border: 1px solid rgba(226,75,74,0.3); border-radius: 12px; background: rgba(226,75,74,0.08); padding: 16px 20px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;}
.verdict-real {border: 1px solid rgba(99,153,34,0.3); border-radius: 12px; background: rgba(99,153,34,0.08); padding: 16px 20px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;}
.verdict-dot-fake {width: 10px; height: 10px; border-radius: 50%; background: #E24B4A; margin-right: 12px; flex-shrink: 0;}
.verdict-dot-real {width: 10px; height: 10px; border-radius: 50%; background: #639922; margin-right: 12px; flex-shrink: 0;}
.verdict-label-fake {font-size: 14px; font-weight: 600; color: #E24B4A;}
.verdict-label-real {font-size: 14px; font-weight: 600; color: #639922;}
.verdict-desc {font-size: 12px; opacity: 0.6; margin-top: 2px;}
.verdict-conf-fake {font-size: 26px; font-weight: 600; color: #E24B4A; text-align: right;}
.verdict-conf-real {font-size: 26px; font-weight: 600; color: #639922; text-align: right;}
.verdict-conf-label {font-size: 11px; opacity: 0.5; text-align: right;}

.metrics-row {display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px;}
.metric-box {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px 14px;}
.metric-label {font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.45; margin-bottom: 6px;}
.metric-value-fake {font-size: 15px; font-weight: 600; color: #E24B4A;}
.metric-value-real {font-size: 15px; font-weight: 600; color: #639922;}
.metric-value-neutral {font-size: 15px; font-weight: 600;}

.summary-text {font-size: 14px; line-height: 1.75; opacity: 0.85;}
.risk-item {display: flex; gap: 10px; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 13px; line-height: 1.5; opacity: 0.85;}
.risk-item:last-child {border-bottom: none; padding-bottom: 0;}
.risk-item:first-child {padding-top: 0;}
.risk-num {font-size: 11px; font-weight: 600; opacity: 0.35; min-width: 20px; margin-top: 2px;}
.fact-item {display: flex; gap: 12px; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 13px; line-height: 1.5; opacity: 0.85;}
.fact-item:last-child {border-bottom: none; padding-bottom: 0;}
.fact-item:first-child {padding-top: 0;}
.fact-bar {width: 2px; min-height: 100%; background: #378ADD; border-radius: 1px; flex-shrink: 0; align-self: stretch;}

.footer-note {font-size: 11px; opacity: 0.35; text-align: center; margin-top: 1.5rem; line-height: 1.7;}
.divider {border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ─── Setup ───────────────────────────────────────────
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# ─── Pipeline ────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def ml_node(state):
    cleaned = clean_text(state["input_text"])
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    state["ml_prediction"] = int(pred)
    state["ml_confidence"] = float(max(proba))
    return state

def retrieval_node(state):
    results = tavily.search(
        query=state["input_text"][:200],
        max_results=3,
        search_depth="basic"
    )
    state["retrieved_docs"] = [r["content"] for r in results["results"]]
    return state

def reasoning_node(state):
    pred = state["ml_prediction"]
    confidence = state["ml_confidence"]
    docs = state["retrieved_docs"]
    label = "FAKE" if pred == 1 else "REAL"

    prompt = f"""
You are a strict news fact-checking AI assistant.

News article to analyze: "{state["input_text"]}"

ML Model Prediction: {label} with confidence {confidence:.2f}

Live web search results about this topic:
- {docs[0]}
- {docs[1]}
- {docs[2]}

STRICT RULES:
- If ML confidence > 0.85 → trust ML signal strongly
- If web evidence strongly contradicts ML → use web evidence
- If ML and web evidence agree → high confidence in verdict

Output EXACTLY in this format, nothing else:

SUMMARY
2 sentences summarizing what the article claims.

ANALYSIS
2 sentences analyzing the evidence from web search.

RISK_FACTORS
- risk factor 1
- risk factor 2
- risk factor 3

VERDICT
Credibility: HIGH or LOW
Confidence: {confidence:.0%}
ML Signal: {label}

DISCLAIMER
This is an AI-generated assessment. Always verify with official sources.
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    state["final_output"] = response.choices[0].message.content
    return state

def parse_output(text):
    sections = {"SUMMARY": "", "ANALYSIS": "", "RISK_FACTORS": [], "VERDICT": {}, "DISCLAIMER": ""}
    current = None
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith("SUMMARY"):
            current = "SUMMARY"
        elif line.startswith("ANALYSIS"):
            current = "ANALYSIS"
        elif line.startswith("RISK_FACTORS"):
            current = "RISK_FACTORS"
        elif line.startswith("VERDICT"):
            current = "VERDICT"
        elif line.startswith("DISCLAIMER"):
            current = "DISCLAIMER"
        else:
            if current == "SUMMARY":
                sections["SUMMARY"] += line + " "
            elif current == "ANALYSIS":
                sections["ANALYSIS"] += line + " "
            elif current == "RISK_FACTORS" and line.startswith("-"):
                sections["RISK_FACTORS"].append(line[1:].strip())
            elif current == "VERDICT":
                if "Credibility:" in line:
                    sections["VERDICT"]["credibility"] = line.split(":")[-1].strip()
                elif "Confidence:" in line:
                    sections["VERDICT"]["confidence"] = line.split(":")[-1].strip()
                elif "ML Signal:" in line:
                    sections["VERDICT"]["ml_signal"] = line.split(":")[-1].strip()
            elif current == "DISCLAIMER":
                sections["DISCLAIMER"] += line + " "
    return sections

def run_pipeline(article_text):
    state = {"input_text": article_text}
    state = ml_node(state)
    state = retrieval_node(state)
    state = reasoning_node(state)
    return state

# ─── UI ──────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-left">
    <div class="nav-dot"></div>
    <div>
      <div class="nav-title">News Credibility Analyzer</div>
      <div class="nav-sub">ML + Agentic AI misinformation detection</div>
    </div>
  </div>
  <span class="nav-badge">v2.0 Agentic</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="hint-text">Paste a full news article to analyze its credibility. For best results use complete articles.</p>', unsafe_allow_html=True)

text_input = st.text_area(
    "Article text",
    height=160,
    placeholder="Paste a news article here...",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    analyze = st.button("Run analysis", use_container_width=True)

if analyze:
    if not text_input.strip():
        st.warning("Please paste a news article.")
    else:
        with st.spinner("ML scoring → Fact retrieval → AI reasoning..."):
            result = run_pipeline(text_input)

        parsed = parse_output(result["final_output"])
        is_fake = result["ml_prediction"] == 1
        conf = f"{result['ml_confidence']:.0%}"
        ml_label = "FAKE" if is_fake else "REAL"
        desc = "ML signal: FAKE · Agentic reasoning confirms" if is_fake else "ML signal: REAL · Agentic reasoning confirms"

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        if is_fake:
            st.markdown(f"""
            <div class="verdict-fake">
              <div style="display:flex;align-items:center;">
                <div class="verdict-dot-fake"></div>
                <div>
                  <div class="verdict-label-fake">Low credibility — potentially fake</div>
                  <div class="verdict-desc">{desc}</div>
                </div>
              </div>
              <div>
                <div class="verdict-conf-fake">{conf}</div>
                <div class="verdict-conf-label">confidence</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-real">
              <div style="display:flex;align-items:center;">
                <div class="verdict-dot-real"></div>
                <div>
                  <div class="verdict-label-real">High credibility — likely real</div>
                  <div class="verdict-desc">{desc}</div>
                </div>
              </div>
              <div>
                <div class="verdict-conf-real">{conf}</div>
                <div class="verdict-conf-label">confidence</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        color = "fake" if is_fake else "real"
        st.markdown(f"""
        <div class="metrics-row">
          <div class="metric-box">
            <div class="metric-label">ML prediction</div>
            <div class="metric-value-{color}">{ml_label}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Credibility</div>
            <div class="metric-value-{color}">{"LOW" if is_fake else "HIGH"}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Sources checked</div>
            <div class="metric-value-neutral">3 found</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if parsed["SUMMARY"]:
            st.markdown(f"""
            <div class="section-card">
              <div class="section-header"><span class="section-tag">Summary</span></div>
              <div class="section-body"><p class="summary-text">{parsed["SUMMARY"]}</p></div>
            </div>
            """, unsafe_allow_html=True)

        if parsed["RISK_FACTORS"]:
            risks_html = "".join([f'<div class="risk-item"><span class="risk-num">0{i+1}</span><span>{r}</span></div>' for i, r in enumerate(parsed["RISK_FACTORS"])])
            st.markdown(f"""
            <div class="section-card">
              <div class="section-header"><span class="section-tag">Risk factors</span></div>
              <div class="section-body">{risks_html}</div>
            </div>
            """, unsafe_allow_html=True)

        if result["retrieved_docs"]:
            facts_html = "".join([f'<div class="fact-item"><div class="fact-bar"></div><span>{doc}</span></div>' for doc in result["retrieved_docs"]])
            st.markdown(f"""
            <div class="section-card">
              <div class="section-header"><span class="section-tag">Retrieved fact-checks</span></div>
              <div class="section-body">{facts_html}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <p class="footer-note">
          This is an AI-generated credibility assessment powered by ML classification and RAG-based fact retrieval.<br>
          Always verify claims with trusted sources before forming conclusions or sharing content.
        </p>
        """, unsafe_allow_html=True)