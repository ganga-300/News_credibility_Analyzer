from graph.state import AgentState
from utils.helpers import get_groq_client
from utils.prompts import VERIFICATION_PROMPT
from utils.parser import parse_json_response

def verification_node(state: AgentState):
    """Verifies each claim against filtered evidence."""
    print(">>> verification_node loaded")
    print(">>> filtered_evidence keys:", list(state.get("filtered_evidence", {}).keys()))

    client = get_groq_client()
    verification_results = []
    
    for claim, docs_objs in state.get("filtered_evidence", {}).items():
        if not docs_objs:
            verification_results.append({
                "claim": claim,
                "status": "INSUFFICIENT_EVIDENCE",
                "confidence": 1.0,
                "reason": "No relevant supporting evidence found after semantic filtering.",
                "evidence": []
            })
            continue

        docs = [d["evidence"] for d in docs_objs]
        docs_text = "\n".join([f"- {d}" for d in docs])
        prompt = VERIFICATION_PROMPT.replace("{{claim}}", claim).replace("{{evidence}}", docs_text)

        # Add these so we can see exactly what the LLM receives
        print(f">>> CLAIM: {claim}")
        print(f">>> DOCS COUNT: {len(docs_objs)}")
        print(f">>> PROMPT:\n{prompt}")
        
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            raw = response.choices[0].message.content
            print(f">>> LLM RAW RESPONSE: {raw}")

            result = parse_json_response(raw, {
                "status": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "reason": "Could not verify."
            })
            
            status = result.get("status", "INSUFFICIENT_EVIDENCE").upper()
            if status not in ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]:
                status = "INSUFFICIENT_EVIDENCE"
                
            verification_results.append({
                "claim": claim,
                "status": status,
                "confidence": float(result.get("confidence", 0.0)),
                "reason": result.get("reason", "No reason provided."),
                "evidence": docs_objs
            })

        except Exception as e:
            print(f">>> Verification failed for claim '{claim}': {e}")
            verification_results.append({
                "claim": claim,
                "status": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "reason": f"Error during verification: {e}",
                "evidence": docs_objs
            })
            
    state["verification_results"] = verification_results
    return state