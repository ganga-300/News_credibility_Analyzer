from graph.state import AgentState

def filtering_node(state: AgentState):
    """
    Applies a strict threshold filter to remove irrelevant evidence.
    """
    filtered = {}
    THRESHOLD = 0.70

    for claim, scored_docs in state.get("ranked_evidence", {}).items():
        valid_docs = []
        for doc_obj in scored_docs:
            if doc_obj.get("similarity", 0.0) >= THRESHOLD:  # ← was "similarity"
                valid_docs.append(doc_obj)

        filtered[claim] = valid_docs

    state["filtered_evidence"] = filtered
    return state