# ============================================================
# GLOBAL RULES (INLINE ONLY — NO .format PLACEHOLDER)
# ============================================================

GLOBAL_RULES = """
GLOBAL RULES (Apply to all reasoning):
- Do not assume facts from tone or sentiment.
- Negative news can still be TRUE.
- Do not generalize (partial truth != full truth).
- Always evaluate the full claim, not partial meaning.
- Be logically consistent.
"""

# ============================================================
# DOMAIN DETECTION PROMPT (NEW)
# ============================================================

DOMAIN_DETECTION_PROMPT = """
Classify the following news article into exactly ONE domain.

Choose from:
- political
- health
- science
- financial
- sports
- entertainment
- general

Article (first 400 chars):
"{text}"

Return ONLY valid JSON, no explanation, no markdown:
{{"domain": "<domain>", "confidence": <0.0-1.0>}}
"""

# ============================================================
# CLAIM EXTRACTION PROMPT
# ============================================================

CLAIM_EXTRACTION_PROMPT = """
Extract ONLY the main verifiable factual claims from the following text.
Extract ONLY statements the article is asserting as true.

DO NOT extract:
- opinions
- predictions
- speculation
- emotional language
- corrections or denials

Break into atomic claims (keep full meaning intact).

Each claim must be on a new line starting with "-".

Text:
"{text}"
"""

# ============================================================
# QUERY GENERATION PROMPT
# ============================================================

QUERY_GENERATION_PROMPT = """
Generate 3 to 5 web search queries for the given claim.

Claim:
"{claim}"

Return JSON ONLY:
{
    "queries": ["query1", "query2", "query3"]
}
"""

# ============================================================
# VERIFICATION PROMPT
# ============================================================

VERIFICATION_PROMPT = """
You are a strict fact-checker.

RULES:
- SUPPORTED: evidence supports claim
- CONTRADICTED: evidence directly disagrees
- INSUFFICIENT_EVIDENCE: no relevant evidence

Claim:
"{{claim}}"

Evidence:
{{evidence}}

Return JSON ONLY:
{
    "status": "SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE",
    "confidence": 0.85,
    "reason": "short explanation"
}
"""

# ============================================================
# REASONING PROMPT
# ============================================================

REASONING_PROMPT = """
GLOBAL RULES:
- Do not assume facts from tone or sentiment.
- Negative news can still be TRUE.
- Do not generalize (partial truth != full truth).
- Always evaluate full claim context.
- Be logically consistent.

INPUT ARTICLE:
{input_text}

ML PREDICTION:
{ml_label} (Confidence: {ml_confidence})

EVIDENCE DATA:
{analysis_data}

Return JSON ONLY:
{{
    "SUMMARY": "2 sentence summary of the article",
    "ANALYSIS": "2-3 sentences comparing ML vs evidence",
    "RISK_FACTORS": [
        "risk factor 1",
        "risk factor 2",
        "risk factor 3"
    ],
    "DISCLAIMER": "This is an AI-generated assessment. Always verify with official sources."
}}
"""

# ============================================================
# LIGHTWEIGHT VERIFICATION PROMPT
# ============================================================

LIGHTWEIGHT_VERIFICATION_PROMPT = """
Quick consistency check.

Text:
"{input_text}"

Return JSON ONLY:
{{
    "verdict": "MOSTLY TRUE | FALSE | OPINION",
    "summary": "1-2 sentence explanation"
}}
"""