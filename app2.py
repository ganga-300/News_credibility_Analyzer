import streamlit as st
import joblib
import string
import nltk
import faiss
import numpy as np
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from groq import Groq

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
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ─── Documents ───────────────────────────────────────
documents = [
    "Vaccines do not cause autism. A study of 650,000 children found no link between MMR vaccine and autism.",
    "COVID-19 vaccines do not contain microchips. This claim has been debunked by WHO and CDC.",
    "Bill Gates is not using vaccines to implant tracking devices. This is a conspiracy theory with no evidence.",
    "5G towers do not spread COVID-19. Viruses cannot travel on radio waves or mobile networks.",
    "Drinking bleach or disinfectant does not cure COVID-19. It is extremely dangerous and can be fatal.",
    "Ivermectin is not a proven cure for COVID-19. WHO and FDA have not approved it for COVID treatment.",
    "Face masks do not cause oxygen deprivation. Medical studies confirm masks are safe to wear.",
    "The 2020 US Presidential Election was not stolen. Over 60 courts including Supreme Court found no evidence of fraud.",
    "Obama was born in Hawaii and is a natural born US citizen. His birth certificate has been verified.",
    "George Soros is not funding antifa or controlling world governments. This is an antisemitic conspiracy theory.",
    "Hillary Clinton did not run a child trafficking ring from a pizza restaurant. This Pizzagate claim is completely false.",
    "Politicians voting on legislation is public record and can be verified through official government websites.",
    "Climate change is real and caused by human activity. 97% of climate scientists agree on this consensus.",
    "The Earth is not flat. This has been proven by centuries of scientific observation and space exploration.",
    "Evolution is a scientific fact supported by fossil records, genetics, and direct observation.",
    "The moon landing in 1969 was real. Thousands of NASA employees and independent scientists confirmed it.",
    "Chemtrails conspiracy is false. Aircraft contrails are water vapor condensation not chemical spraying.",
    "Celebrity death rumors spread rapidly on social media and are often completely fabricated.",
    "Fake quotes attributed to celebrities and politicians are common misinformation tactics.",
    "Deepfake videos can make celebrities appear to say things they never said.",
    "Celebrity health rumors are frequently exaggerated or completely false on social media.",
    "Economic statistics should be verified through official sources like World Bank or IMF.",
    "Cryptocurrency investment scams promising guaranteed returns are fraudulent.",
    "Get rich quick schemes promoted on social media are almost always scams.",
    "Stock market predictions claiming guaranteed profits are misleading and potentially fraudulent.",
    "Sensational headlines designed to provoke emotional reactions are a common misinformation tactic.",
    "News articles without author names or credible sources should be treated with suspicion.",
    "Misinformation often spreads through emotional manipulation rather than factual evidence.",
    "Satire websites are sometimes mistaken for real news causing widespread misinformation.",
    "Out of context images and videos are frequently used to spread false narratives online."
]

doc_embeddings = embed_model.encode(documents)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

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

from tavily import TavilyClient

tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

def retrieval_node(state):
    # Search the live internet instead of fixed docs
    results = tavily.search(
        query=state["input_text"][:200],
        max_results=3,
        search_depth="basic"
    )
    
    docs = [r["content"] for r in results["results"]]
    state["retrieved_docs"] = docs
    return state

def reasoning_node(state):
    pred = state["ml_prediction"]
    confidence = state["ml_confidence"]
    docs = state["retrieved_docs"]
    label = "FAKE" if pred == 1 else "REAL"

    prompt = f"""
You are a strict fact-checking AI assistant.

User claim: "{state["input_text"]}"

ML Prediction: {label} with confidence {confidence:.2f}
NOTE: ML model is trained on news articles. For short factual claims, 
trust the web search evidence MORE than the ML prediction.

Live web search results:
- {docs[0]}
- {docs[1]}
- {docs[2]}

STRICT RULES:
- If web evidence CONFIRMS the claim → Credibility HIGH regardless of ML
- If web evidence CONTRADICTS the claim → Credibility LOW
- If ML confidence > 0.9 AND evidence agrees → strongly LOW
- Short factual claims: trust evidence over ML signal

Output EXACTLY:

SUMMARY
2 sentences about what the claim states.

ANALYSIS  
2 sentences based on web evidence found.

RISK_FACTORS
- factor 1
- factor 2
- factor 3

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

# def run_pipeline(article_text):
#     state = {"input_text": article_text}
#     state = ml_node(state)
#     state = retrieval_node(state)
#     state = reasoning_node(state)
#     return state

def run_pipeline(article_text):
    state = {"input_text": article_text}
    
    # Ask LLM to classify input type first
    classify_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"""
Is this a news article or a short factual claim?
Text: "{article_text[:300]}"
Reply with ONLY one word: ARTICLE or CLAIM
"""}]
    )
    
    input_type = classify_response.choices[0].message.content.strip().upper()
    state["input_type"] = input_type
    
    # Only run ML if it's a news article
    if "ARTICLE" in input_type:
        state = ml_node(state)
    else:
        state["ml_prediction"] = -1
        state["ml_confidence"] = 0.0
    
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

text_input = st.text_area(
    "Article text",
    height=160,
    placeholder="Paste a news article here to analyze its credibility...",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    analyze = st.button("Run analysis", use_container_width=True)

if analyze:
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("ML scoring → Fact retrieval → AI reasoning..."):
            result = run_pipeline(text_input)

        parsed = parse_output(result["final_output"])
        is_fake = result["ml_prediction"] == 1
        conf = f"{result['ml_confidence']:.0%}"
        ml_label = "FAKE" if is_fake else "REAL"

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Verdict banner
        if is_fake:
            st.markdown(f"""
            <div class="verdict-fake">
              <div style="display:flex;align-items:center;">
                <div class="verdict-dot-fake"></div>
                <div>
                  <div class="verdict-label-fake">Low credibility — potentially fake</div>
                  <div class="verdict-desc">ML signal: FAKE · Agentic reasoning confirms</div>
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
                  <div class="verdict-desc">ML signal: REAL · Agentic reasoning confirms</div>
                </div>
              </div>
              <div>
                <div class="verdict-conf-real">{conf}</div>
                <div class="verdict-conf-label">confidence</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Metrics
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

        # Summary
        if parsed["SUMMARY"]:
            st.markdown(f"""
            <div class="section-card">
              <div class="section-header"><span class="section-tag">Summary</span></div>
              <div class="section-body"><p class="summary-text">{parsed["SUMMARY"]}</p></div>
            </div>
            """, unsafe_allow_html=True)

        # Risk factors
        if parsed["RISK_FACTORS"]:
            risks_html = "".join([f'<div class="risk-item"><span class="risk-num">0{i+1}</span><span>{r}</span></div>' for i, r in enumerate(parsed["RISK_FACTORS"])])
            st.markdown(f"""
            <div class="section-card">
              <div class="section-header"><span class="section-tag">Risk factors</span></div>
              <div class="section-body">{risks_html}</div>
            </div>
            """, unsafe_allow_html=True)

        # Fact checks
        facts_html = "".join([f'<div class="fact-item"><div class="fact-bar"></div><span>{doc}</span></div>' for doc in result["retrieved_docs"]])
        st.markdown(f"""
        <div class="section-card">
          <div class="section-header"><span class="section-tag">Retrieved fact-checks</span></div>
          <div class="section-body">{facts_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <p class="footer-note">
          This is an AI-generated credibility assessment powered by ML classification and RAG-based fact retrieval.<br>
          Always verify claims with trusted sources before forming conclusions or sharing content.
        </p>
        """, unsafe_allow_html=True)