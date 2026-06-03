# Agentic News Credibility Analyzer

![Version](https://img.shields.io/badge/version-5.0%20Agentic-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-enabled-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-yellow)

A portfolio-quality AI system that goes beyond simple machine learning classification by implementing a modular **LangGraph Agent Architecture**. This system acts as a multi-step fact-checker, employing a Retrieval-Augmented Generation (RAG) pipeline to extract atomic claims, retrieve live web evidence, perform strict semantic verification, and logically aggregate verdicts.

## 🚀 Features

- **Multi-Agent Orchestration (LangGraph):** Employs specialized nodes for preprocessing, claim extraction, semantic ranking, and logical aggregation.
- **Dynamic Routing:** Implements a "Confidence Router" that fast-tracks high-confidence ML predictions while delegating low-confidence inputs to a deep RAG verification pipeline.
- **Atomic Claim Extraction:** Uses Groq LLMs to intelligently break down complex articles into isolated, verifiable assertions while stripping out caveats and hedges.
- **Evidence-Based RAG Verification:** Leverages the Tavily Search API to dynamically source live evidence and evaluate claims strictly against retrieved data, avoiding AI hallucinations.
- **Deterministic Aggregation:** Prevents LLM "rationalization" by enforcing strict boolean logic on the final verdict based on verified claims (e.g., a single FALSE claim downgrades the overall score).
- **Professional Streamlit UI:** A clean, responsive dashboard that visualizes the AI's internal reasoning, claim-by-claim analysis, and identified risk factors.

## 🧠 System Architecture

The system utilizes a StateGraph to pass a strongly typed `AgentState` through multiple reasoning nodes. 

For a detailed visual breakdown and Mermaid diagram, see the [Architecture Document](architecture.md).

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- [Groq API Key](https://console.groq.com/)
- [Tavily API Key](https://app.tavily.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/news-credibility-analyzer.git
   cd news-credibility-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets**
   Create a `.streamlit/secrets.toml` file in the root directory:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   TAVILY_API_KEY = "your_tavily_api_key_here"
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## ☁️ Streamlit Cloud Deployment

This project is structured specifically to be deployable on Streamlit Community Cloud. 
1. Push this repository to GitHub.
2. Log into [Streamlit Cloud](https://share.streamlit.io/).
3. Create a new app, select your repository, and set the main file path to `app.py`.
4. In the Advanced Settings, paste your API keys into the Secrets section using the same TOML format shown above.

# News Credibility Analyzer

An AI-powered misinformation detection system that combines classical Machine Learning with Agentic AI to analyze news articles and generate structured credibility reports.

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

