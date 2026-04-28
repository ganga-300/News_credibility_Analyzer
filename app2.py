import streamlit as st
import joblib
import string
import nltk
import faiss
import numpy as np
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─── Setup ───────────────────────────────────────────
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load Phase 1 model
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ─── Fact-check Documents ────────────────────────────
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

# Build FAISS index
doc_embeddings = embed_model.encode(documents)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# ─── Helper Functions ────────────────────────────────
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
    query_embedding = embed_model.encode([state["input_text"]])
    distances, indices = index.search(np.array(query_embedding), k=3)
    state["retrieved_docs"] = [documents[i] for i in indices[0]]
    return state

def reasoning_node(state):
    pred = state["ml_prediction"]
    confidence = state["ml_confidence"]
    docs = state["retrieved_docs"]
    label = "FAKE" if pred == 1 else "REAL"

    prompt = f"""
You are a strict fact-checking AI assistant.

ML Prediction: {label} with confidence {confidence:.2f}

Evidence from fact-check database:
- {docs[0]}
- {docs[1]}
- {docs[2]}

Output EXACTLY in this format, nothing else:

📋 SUMMARY
Write 2 sentences summarizing what the article claims.

🔍 ANALYSIS
Write 2 sentences analyzing the evidence found.

⚠️ RISK FACTORS
List 3 specific red flags found in this article.

✅ VERDICT
Credibility: HIGH or LOW
Confidence: {confidence:.0%}
ML Signal: {label}

⚠️ DISCLAIMER
This is an AI-generated assessment. Always verify with official sources.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    state["final_output"] = response.choices[0].message.content
    return state

def run_pipeline(article_text):
    state = {"input_text": article_text}
    state = ml_node(state)
    state = retrieval_node(state)
    state = reasoning_node(state)
    return state

# ─── Streamlit UI ────────────────────────────────────
st.set_page_config(page_title="News Credibility Agent", page_icon="🤖")

st.title("🤖 Agentic News Credibility Analyzer")
st.write("AI-powered misinformation detection with fact-checking and reasoning.")

st.markdown("---")

text_input = st.text_area(
    "📰 Paste Article Text",
    height=250,
    placeholder="Paste a news article here..."
)

if st.button("🔍 Analyze with AI Agent", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("🤖 Agent is analyzing... (ML scoring → Fact retrieval → AI reasoning)"):
            result = run_pipeline(text_input)

        st.markdown("---")
        st.subheader("📊 Agent Report")
        st.markdown(result["final_output"])

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ML Confidence", f"{result['ml_confidence']:.0%}")
        with col2:
            label = "🚨 FAKE" if result['ml_prediction'] == 1 else "✅ REAL"
            st.metric("ML Signal", label)

        with st.expander("🔎 Retrieved Fact-Checks"):
            for i, doc in enumerate(result['retrieved_docs']):
                st.write(f"**{i+1}.** {doc}")