from dotenv import load_dotenv
load_dotenv()



from typing import TypedDict, List
import string
import os
from dotenv import load_dotenv
load_dotenv()
import nltk
import joblib
from nltk.corpus import stopwords
from tavily import TavilyClient
from groq import Groq
from langgraph.graph import StateGraph, START, END

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "models", "tfidf.pkl"))

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

class NewsState(TypedDict):
    input_text: str
    ml_prediction: int
    ml_confidence: float
    retrieved_docs: List[str]
    evidence_score: float
    final_output: str

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def ml_classifier(state: NewsState):
    cleaned_text = clean_text(state["input_text"])
    vector = tfidf.transform([cleaned_text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = float(max(probabilities))
    return {"ml_prediction": int(prediction), "ml_confidence": confidence}

def confidence_router(state: NewsState):
    if state["ml_confidence"] >= 0.90:
        return "high_confidence"
    return "low_confidence"

def fast_verdict(state: NewsState):
    ml_label = "FAKE" if state["ml_prediction"] == 1 else "REAL"
    credibility = "LOW" if state["ml_prediction"] == 1 else "HIGH"
    output = f"""SUMMARY
Prediction generated directly from the ML classifier.

ANALYSIS
The model confidence exceeded the routing threshold of 90%. External retrieval and LLM reasoning were skipped.

RISK_FACTORS
- No live evidence retrieval performed
- Decision based on historical training data
- Model confidence triggered shortcut path

VERDICT
Credibility: {credibility}
Confidence: {state['ml_confidence']:.0%}
ML Signal: {ml_label}

DISCLAIMER
This is an AI-generated assessment. Verify information using trusted sources."""
    return {"final_output": output}

def retrieval_node(state: NewsState):
    results = tavily_client.search(query=state["input_text"][:200], max_results=5, search_depth="advanced")
    docs = [item.get("content", "") for item in results.get("results", [])]
    return {"retrieved_docs": docs}

def evidence_check(state: NewsState):
    docs = state["retrieved_docs"]
    score = 0
    positive_terms = ["verified", "confirmed", "official", "true", "accurate"]
    negative_terms = ["false", "fake", "misleading", "debunked", "hoax"]
    for doc in docs:
        lower_doc = doc.lower()
        for word in positive_terms:
            if word in lower_doc: score += 1
        for word in negative_terms:
            if word in lower_doc: score -= 1
    return {"evidence_score": float(score)}

def llm_judge(state: NewsState):
    ml_label = "FAKE" if state["ml_prediction"] == 1 else "REAL"
    docs = state["retrieved_docs"] if state["retrieved_docs"] else ["No evidence found"]
    evidence_text = "\n".join([f"- {doc}" for doc in docs])
    prompt = f"""You are a professional news fact-checking AI.

ARTICLE
{state["input_text"]}

ML PREDICTION
{ml_label}

ML CONFIDENCE
{state["ml_confidence"]:.2f}

EVIDENCE SCORE
{state["evidence_score"]}

WEB EVIDENCE
{evidence_text}

RULES
1. If ML confidence is high, consider ML strongly.
2. If web evidence contradicts ML, prioritize evidence.
3. If both agree, confidence should be high.
4. Be objective and evidence-driven.

OUTPUT EXACTLY IN THIS FORMAT

SUMMARY
2 sentences summarizing the claim.

ANALYSIS
2 sentences analyzing retrieved evidence.

RISK_FACTORS
- factor 1
- factor 2
- factor 3

VERDICT
Credibility: HIGH or LOW
Confidence: XX%
ML Signal: {ml_label}

DISCLAIMER
This is an AI-generated assessment. Always verify with official sources."""

    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"final_output": response.choices[0].message.content}

builder = StateGraph(NewsState)
builder.add_node("ml_classifier", ml_classifier)
builder.add_node("fast_verdict", fast_verdict)
builder.add_node("retrieval_node", retrieval_node)
builder.add_node("evidence_check", evidence_check)
builder.add_node("llm_judge", llm_judge)
builder.add_edge(START, "ml_classifier")
builder.add_conditional_edges("ml_classifier", confidence_router, {"high_confidence": "fast_verdict", "low_confidence": "retrieval_node"})
builder.add_edge("retrieval_node", "evidence_check")
builder.add_edge("evidence_check", "llm_judge")
builder.add_edge("fast_verdict", END)
builder.add_edge("llm_judge", END)
graph = builder.compile()

def analyze_news(article_text: str):
    return graph.invoke({"input_text": article_text})
