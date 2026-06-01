import json
from graph.state import AgentState
from utils.helpers import get_groq_client
from utils.prompts import REASONING_PROMPT
from utils.parser import parse_json_response


def reasoning_node(state: AgentState):
    """Generates final reasoning summary from ML + verification results."""

    client = get_groq_client()

    # -----------------------------
    # ML LABEL
    # -----------------------------
    ml_pred = state.get("ml_prediction", -1)
    ml_label = "FAKE" if ml_pred == 1 else "REAL" if ml_pred == 0 else "UNKNOWN"

    # -----------------------------
    # VERIFICATIONS
    # -----------------------------
    verifications = state.get("verification_results", [])

    if not verifications:
        state["reasoning_summary"] = {
            "SUMMARY": "No factual claims were verified.",
            "ANALYSIS": "Pipeline had no usable evidence.",
            "RISK_FACTORS": [],
            "DISCLAIMER": "Always verify with trusted sources."
        }
        return state

    # -----------------------------
    # SAFE JSON ENCODING
    # -----------------------------
    analysis_data = json.dumps([
        {
            "claim": v.get("claim", ""),
            "status": v.get("status", ""),
            "reason": v.get("reason", "")
        }
        for v in verifications
    ])

    # -----------------------------
    # BUILD PROMPT (SAFE)
    # -----------------------------
    prompt = REASONING_PROMPT.format(
        input_text=state["input_text"][:1000],
        ml_label=ml_label,
        ml_confidence=f"{state.get('ml_confidence', 0):.2f}",
        analysis_data=analysis_data
    )

    print("\n" + "=" * 60)
    print("REASONING PROMPT")
    print("=" * 60)
    print(prompt)

    # -----------------------------
    # CALL LLM
    # -----------------------------
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        result = parse_json_response(
            response.choices[0].message.content,
            {}
        )

        state["reasoning_summary"] = {
            "SUMMARY": result.get("SUMMARY", "Summary unavailable."),
            "ANALYSIS": result.get("ANALYSIS", "Analysis unavailable."),
            "RISK_FACTORS": result.get("RISK_FACTORS", []),
            "DISCLAIMER": result.get(
                "DISCLAIMER",
                "Always verify with official sources."
            )
        }

    except Exception as e:
        print("\nREASONING ERROR:", str(e))

        state["reasoning_summary"] = {
            "SUMMARY": "Error generating summary.",
            "ANALYSIS": "Error generating analysis.",
            "RISK_FACTORS": [],
            "DISCLAIMER": "Always verify with official sources."
        }

    return state