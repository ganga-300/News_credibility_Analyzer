import joblib
import os
from graph.state import AgentState

# Load models safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
TFIDF_PATH = os.path.join(BASE_DIR, 'models', 'tfidf.pkl')

try:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
except Exception as e:
    # If models can't load, we handle it gracefully or raise
    model = None
    tfidf = None

def classification_node(state: AgentState):
    """Generates an ML prediction and confidence score."""
    if not model or not tfidf:
        state["ml_prediction"] = -1
        state["ml_confidence"] = 0.0
        return state
        
    cleaned = state.get("cleaned_text", state["input_text"])
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    
    state["ml_prediction"] = int(pred)
    state["ml_confidence"] = float(max(proba))
    return state
