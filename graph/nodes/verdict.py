from graph.state import AgentState


def verdict_node(state: AgentState):
    """
    Computes final credibility score.

    ML weight is only applied for political domain.
    For all other domains, ML score is zeroed out and
    verdict relies entirely on verification + evidence.
    """
    verifications = state.get("verification_results", [])
    ml_confidence = state.get("ml_confidence", 0.0)
    ml_prediction = state.get("ml_prediction", -1)
    domain        = state.get("domain", "general")

    if not verifications:
        state["credibility_score"] = "UNCERTAIN"
        return state

    # ML score
    if ml_prediction == 0:
        ml_score = ml_confidence
    elif ml_prediction == 1:
        ml_score = 1.0 - ml_confidence
    else:
        ml_score = 0.5

    # Verification score
    supported_count    = sum(1 for v in verifications if v["status"] == "SUPPORTED")
    contradicted_count = sum(1 for v in verifications if v["status"] == "CONTRADICTED")
    insufficient_count = sum(1 for v in verifications if v["status"] == "INSUFFICIENT_EVIDENCE")
    total_verifications = len(verifications)

    verif_score = supported_count / total_verifications if total_verifications > 0 else 0.5

    # Evidence similarity score
    total_sim = 0.0
    total_evidence_pieces = 0

    for v in verifications:
        for doc in v.get("evidence", []):
            total_sim += doc.get("similarity", 0.0)
            total_evidence_pieces += 1

    evidence_score = total_sim / total_evidence_pieces if total_evidence_pieces > 0 else 0.0

    # ── DOMAIN-AWARE WEIGHT CALCULATION ──────────────────────
    #
    # Political domain: ML model was trained here, trust it
    # All other domains: ML model has no knowledge, zero it out
    #
    if domain == "political":
        # Original dynamic weighting based on ML reliability
        ml_reliability       = abs(ml_confidence - 0.5) * 2   # 0.0 at 50%, 1.0 at 100%
        effective_ml_weight  = 0.4 * ml_reliability
        effective_verif_weight   = 0.4 + (0.4 - effective_ml_weight) * 0.5
        effective_evidence_weight = 1.0 - effective_ml_weight - effective_verif_weight
        print(f"VERDICT: Using ML weight={effective_ml_weight:.2f} (political domain)")
    else:
        # Non-political: ML contributes nothing
        effective_ml_weight       = 0.0
        effective_verif_weight    = 0.6
        effective_evidence_weight = 0.4
        print(f"VERDICT: ML weight=0.0 (domain={domain}, ML not trusted)")

    # ── FINAL SCORE ──────────────────────────────────────────
    if total_evidence_pieces == 0:
        if contradicted_count > 0:
            final_score = 0.1
        elif insufficient_count == total_verifications:
            # All insufficient
            if domain == "political":
                final_score = ml_score   # fall back to ML for political
            else:
                final_score = 0.5        # genuinely uncertain for others
        else:
            final_score = 0.5
    else:
        final_score = (
            (effective_ml_weight       * ml_score)      +
            (effective_verif_weight    * verif_score)   +
            (effective_evidence_weight * evidence_score)
        )

    # ── CREDIBILITY LABEL ────────────────────────────────────
    if final_score >= 0.75:
        credibility = "TRUE"
    elif final_score >= 0.60:
        credibility = "MOSTLY TRUE"
    elif final_score >= 0.45:
        credibility = "MIXED"
    elif final_score >= 0.30:
        credibility = "MOSTLY FALSE"
    else:
        credibility = "FALSE"

    if contradicted_count > 0 and final_score >= 0.45:
        credibility = "MIXED"

    print(f"VERDICT: domain={domain.upper()} | final_score={final_score:.2f} | credibility={credibility}")

    state["credibility_score"] = credibility
    return state