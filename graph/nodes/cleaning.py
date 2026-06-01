import re
from graph.state import AgentState

def cleaning_node(state: AgentState):
    """
    Cleans the retrieved evidence by removing boilerplate,
    excessive whitespace, and empty strings.
    """
    cleaned = {}
    
    for claim, docs in state.get("retrieved_evidence", {}).items():
        clean_docs = []
        for doc in docs:
            # FIX 1: extract text from dict, was treating dict as string
            raw_text = doc.get("content", "") if isinstance(doc, dict) else doc

            # FIX 2: was r'\\s+' (escaped backslash, never matched), should be r'\s+'
            text = re.sub(r'\s+', ' ', raw_text).strip()
            
            if len(text) > 30:
                clean_docs.append(text)
                
        cleaned[claim] = clean_docs
        
    state["cleaned_evidence"] = cleaned
    return state