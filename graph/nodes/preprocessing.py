import os
import re
import json
import string
import nltk
from nltk.corpus import stopwords
from groq import Groq
from graph.state import AgentState
from utils.prompts import DOMAIN_DETECTION_PROMPT

# Ensure stopwords are downloaded
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)


def detect_domain(text: str) -> str:
    """Uses Groq LLM to detect the news domain."""
    try:
        prompt = DOMAIN_DETECTION_PROMPT.format(text=text[:400])
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        domain = data.get("domain", "general").lower()

        valid_domains = {"political", "health", "science", "financial", "sports", "entertainment", "general"}
        return domain if domain in valid_domains else "general"

    except Exception:
        return "general"


def preprocessing_node(state: AgentState):
    print("\n" + "=" * 50)
    print("INPUT ARTICLE")
    print(state["input_text"])
    print("=" * 50)

    # Clean text for ML
    state["cleaned_text"] = clean_text(state["input_text"])

    # Detect domain
    domain = detect_domain(state["input_text"])
    state["domain"] = domain

    print(f"DETECTED DOMAIN: {domain.upper()}")
    print("=" * 50)

    return state