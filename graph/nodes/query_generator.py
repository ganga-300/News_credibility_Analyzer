import json
from graph.state import AgentState
from utils.helpers import get_groq_client
from utils.prompts import QUERY_GENERATION_PROMPT
from utils.parser import parse_json_response

def query_generator_node(state: AgentState):
    """
    Generates optimized search queries for each extracted claim.
    """
    client = get_groq_client()
    generated_queries = {}
    
    for claim in state.get("extracted_claims", [])[:5]:
        prompt = QUERY_GENERATION_PROMPT.replace("{claim}", claim)
        
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            result = parse_json_response(response.choices[0].message.content, {"queries": []})
            queries = result.get("queries", [])
            
            # Fallback to claim itself if generation fails or is empty
            if not queries or not isinstance(queries, list):
                queries = [claim]
                
            generated_queries[claim] = queries[:5] # Max 5 queries per claim
        except Exception:
            generated_queries[claim] = [claim]
            
    state["generated_queries"] = generated_queries
    return state
