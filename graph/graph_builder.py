from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes.preprocessing import preprocessing_node
from graph.nodes.classifier import classification_node
from graph.nodes.lightweight import lightweight_verification_node
from graph.nodes.claim_extractor import extract_claims_node
from graph.nodes.query_generator import query_generator_node
from graph.nodes.retrieval import retrieval_node
from graph.nodes.cleaning import cleaning_node
from graph.nodes.scoring import scoring_node
from graph.nodes.ranking import ranking_node
from graph.nodes.filtering import filtering_node
from graph.nodes.verification import verification_node
from graph.nodes.reasoning import reasoning_node
from graph.nodes.verdict import verdict_node


def route_by_confidence(state: AgentState):
    """
    Domain-aware router.

    Logic:
    - POLITICAL + confidence > 0.85  → lightweight_verification_node (ML is reliable, fast path)
    - POLITICAL + confidence <= 0.85 → extract_claims_node (ML unsure, deep verify)
    - ANY OTHER DOMAIN               → extract_claims_node (ML not trained here, always deep verify)
    """
    confidence = state.get("ml_confidence", 0.0)
    domain = state.get("domain", "general")

    if domain == "political" and confidence > 0.85:
        print(f"ROUTING: Fast path (political, confidence={confidence:.2f})")
        return "lightweight_verification_node"

    print(f"ROUTING: Deep verify path (domain={domain}, confidence={confidence:.2f})")
    return "extract_claims_node"


def build_graph():
    """Compiles and returns the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("preprocessing_node",             preprocessing_node)
    workflow.add_node("classification_node",            classification_node)
    workflow.add_node("lightweight_verification_node",  lightweight_verification_node)
    workflow.add_node("extract_claims_node",            extract_claims_node)
    workflow.add_node("query_generator_node",           query_generator_node)
    workflow.add_node("retrieval_node",                 retrieval_node)

    # Semantic Pipeline
    workflow.add_node("cleaning_node",                  cleaning_node)
    workflow.add_node("scoring_node",                   scoring_node)
    workflow.add_node("ranking_node",                   ranking_node)
    workflow.add_node("filtering_node",                 filtering_node)

    workflow.add_node("verification_node",              verification_node)
    workflow.add_node("reasoning_node",                 reasoning_node)
    workflow.add_node("verdict_node",                   verdict_node)

    # Build Edges
    workflow.set_entry_point("preprocessing_node")
    workflow.add_edge("preprocessing_node", "classification_node")

    # Conditional Routing (domain-aware)
    workflow.add_conditional_edges("classification_node", route_by_confidence)

    # Fast Track Path (political only)
    workflow.add_edge("lightweight_verification_node", "verdict_node")

    # Deep Verification Path (all non-political + low confidence political)
    workflow.add_edge("extract_claims_node",    "query_generator_node")
    workflow.add_edge("query_generator_node",   "retrieval_node")
    workflow.add_edge("retrieval_node",         "cleaning_node")
    workflow.add_edge("cleaning_node",          "scoring_node")
    workflow.add_edge("scoring_node",           "ranking_node")
    workflow.add_edge("ranking_node",           "filtering_node")
    workflow.add_edge("filtering_node",         "verification_node")
    workflow.add_edge("verification_node",      "reasoning_node")
    workflow.add_edge("reasoning_node",         "verdict_node")

    # End
    workflow.add_edge("verdict_node", END)

    return workflow.compile()