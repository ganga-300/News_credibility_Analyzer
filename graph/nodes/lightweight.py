import json
from graph.state import AgentState
from utils.helpers import get_groq_client
from utils.prompts import LIGHTWEIGHT_VERIFICATION_PROMPT
from utils.parser import parse_json_response

def lightweight_verification_node(state: AgentState):
    """Performs a quick LLM check for high-confidence predictions."""
    client = get_groq_client()
    
    ml_pred = state.get("ml_prediction", -1)
    label = "FAKE" if ml_pred == 1 else "REAL" if ml_pred == 0 else "UNKNOWN"
    
    prompt = LIGHTWEIGHT_VERIFICATION_PROMPT.format(
        ml_label=label,
        input_text=state["input_text"][:2000] # Limit context window
    )
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    result = parse_json_response(response.choices[0].message.content, {
        "verdict": "UNCERTAIN", 
        "summary": "Fallback summary."
    })
    
    state["credibility_score"] = result.get("verdict", "UNCERTAIN")
    state["reasoning_summary"] = {
        "SUMMARY": result.get("summary", ""),
        "ANALYSIS": f"Lightweight verification based on high ML confidence ({state.get('ml_confidence', 0):.0%}).",
        "RISK_FACTORS": [],
        "DISCLAIMER": "High-confidence fast track. Deep retrieval was skipped."
    }
    
    return state
