from graph.state import AgentState

def ranking_node(state: AgentState):
    """
    Ranks evidence strictly by the semantic similarity score computed in the scoring node.
    Keeps only the top 3 most relevant pieces of evidence per claim.
    """
    ranked = {}
    
    for claim, scored_docs in state.get("scored_evidence", {}).items():
        if not scored_docs:
            ranked[claim] = []
            continue
            
        # Sort by similarity score descending
        sorted_docs = sorted(scored_docs, key=lambda x: x.get("similarity", 0.0), reverse=True)
        
        # Keep top 3
        ranked[claim] = sorted_docs[:3]
        
    state["ranked_evidence"] = ranked
    return state
