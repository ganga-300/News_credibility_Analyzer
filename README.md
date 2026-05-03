# News Credibility Analyzer

An AI-powered misinformation detection system that combines classical Machine Learning with Agentic AI to analyze news articles and generate structured credibility reports.

## Live Demo
- Phase 1 — ML Classifier: https://newscredibilityanalyzer-otxyjaxgeydnsbgvnwmwjl.streamlit.app
- Phase 2 — Agentic AI: https://newscredibilityanalyzergit-deyscn4e2fgxait9rj2hjf.streamlit.app/

## Project Overview

This project is built in two phases, progressively evolving from a classical ML classifier to a fully agentic AI system.

### Phase 1 — ML Based Classification
A machine learning pipeline that classifies news articles as credible or fake using NLP techniques.

### Phase 2 — Agentic AI System
An agentic pipeline that autonomously retrieves live fact-checks and generates structured credibility reports using LLM reasoning.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | Scikit-learn, TF-IDF, Logistic Regression |
| LLM | Groq API (Llama 3) |
| Web Search | Tavily Search API |
| UI | Streamlit |
| Language | Python |

---

## How It Works

### Phase 1 Pipeline

Input Article → Text Preprocessing → TF-IDF Vectorization → Logistic Regression → Credibility Score

### Phase 2 Pipeline

Input Article → ML Scoring → Tavily Live Search → LLM Reasoning → Structured Report

---

## Features

**Phase 1**
- Text preprocessing with NLTK stopword removal
- TF-IDF feature extraction
- Logistic Regression classifier trained on 44,000+ articles
- 99% classification accuracy
- Clean Streamlit UI with confidence score display

**Phase 2**
- Agentic pipeline with 3 nodes — ML scoring, fact retrieval, LLM reasoning
- Live web search using Tavily API for real-time fact-checking
- Groq Llama 3 for structured report generation
- Hallucination reduction by grounding LLM in real web sources
- Structured output — Summary, Analysis, Risk Factors, Verdict, Disclaimer

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 99.06% |
| Precision | 99.42% |
| Recall | 98.79% |
| F1 Score | 99.10% |

---

## Project Structure

News_credibility_Analyzer/
├── app.py              ← Phase 1 Streamlit app
├── app2.py             ← Phase 2 Agentic AI app
├── model.pkl           ← Trained ML model
├── tfidf.pkl           ← TF-IDF vectorizer
├── requirements.txt
└── README.md

## Limitations
- ML model performs best on political and health misinformation
- For best results paste complete news articles
- Always verify results with trusted sources

---

## Author

**Ganga Raghuwanshi**
- GitHub: https://github.com/ganga-300
- Email: ganga.raghuwanshi2024@nst.rishihood.edu.in

---

