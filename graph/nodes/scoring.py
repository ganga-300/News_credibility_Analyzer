from graph.state import AgentState

# Lazy load the model to avoid slow startup times
_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer('all-MiniLM-L6-v2')
    return _encoder

def scoring_node(state: AgentState):
    """
    Computes semantic similarity between the claim and the cleaned evidence.
    """
    try:
        encoder = get_encoder()
        from sentence_transformers import util
    except Exception as e:
        # If sentence_transformers fails to load, fallback to naive scoring
        print(f"Warning: Could not load sentence-transformers. ({e})")
        encoder = None
        
    scored = {}
    
    for claim, docs in state.get("cleaned_evidence", {}).items():
        scored_docs = []
        if not docs:
            scored[claim] = []
            continue
            
        if encoder:
            claim_emb = encoder.encode(claim, convert_to_tensor=True)
            doc_embs = encoder.encode(docs, convert_to_tensor=True)
            
            # Compute cosine similarities
            cosine_scores = util.cos_sim(claim_emb, doc_embs)[0]
            
            for i, score in enumerate(cosine_scores):
                scored_docs.append({
                    "evidence": docs[i],
                    "similarity": float(score)
                })
        else:
            # Fallback if ML fails
            for doc in docs:
                scored_docs.append({
                    "evidence": doc,
                    "similarity": 1.0 # assume all valid if model fails
                })
                
        scored[claim] = scored_docs
        
    state["scored_evidence"] = scored
    return state
