from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    input_text: str
    cleaned_text: str

    # Domain Detection (NEW)
    domain: Optional[str]  # "political" | "health" | "science" | "financial" | "sports" | "entertainment" | "general"

    # ML Classification
    ml_prediction: Optional[int]
    ml_confidence: Optional[float]

    # Claim Extraction & Verification
    extracted_claims: List[str]
    generated_queries: Dict[str, List[str]]         # maps claim to list of optimized search queries
    retrieved_evidence: Dict[str, List[str]]        # maps claim to raw evidence strings
    cleaned_evidence: Dict[str, List[str]]          # post-cleaning
    scored_evidence: Dict[str, List[Dict[str, Any]]]  # list of dicts: evidence, similarity
    ranked_evidence: Dict[str, List[Dict[str, Any]]]  # sorted by similarity
    filtered_evidence: Dict[str, List[Dict[str, Any]]]  # post-threshold filtering
    verification_results: List[Dict[str, Any]]      # list of dicts: claim, status, reason, confidence, evidence

    # Final Outputs
    credibility_score: Optional[str]                # MOSTLY TRUE, PARTIALLY TRUE, FALSE, UNCERTAIN, OPINION
    reasoning_summary: Dict[str, Any]               # SUMMARY, ANALYSIS, RISK_FACTORS, DISCLAIMER
    final_output: str                               # combined output string or JSON for UI