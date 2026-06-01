import streamlit as st
import traceback
from graph.graph_builder import build_graph

# ─── Page Config ─────────────────────────────────────
st.set_page_config(page_title="Agentic News Credibility Analyzer", page_icon="🔍", layout="centered")

# ─── Custom CSS ──────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 800px;}

.navbar {display: flex; align-items: center; justify-content: space-between; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 2rem;}
.navbar-left {display: flex; align-items: center; gap: 10px;}
.nav-dot {width: 8px; height: 8px; border-radius: 50%; background: #378ADD;}
.nav-title {font-size: 16px; font-weight: 600; letter-spacing: -0.01em;}
.nav-sub {font-size: 13px; opacity: 0.6; margin-top: 2px;}
.nav-badge {font-size: 11px; padding: 4px 10px; border-radius: 20px; background: rgba(55,138,221,0.15); color: #378ADD; font-weight: 600;}

.hint-text {font-size: 14px; opacity: 0.6; margin-bottom: 1.5rem;}

.section-card {border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; margin-bottom: 16px; background: rgba(255,255,255,0.02);}
.section-header {padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.04); display: flex; justify-content: space-between; align-items: center;}
.section-tag {font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.7;}
.section-body {padding: 16px;}

.verdict-FALSE {border: 1px solid rgba(226,75,74,0.4); border-radius: 12px; background: rgba(226,75,74,0.1); padding: 20px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;}
.verdict-MOSTLY-TRUE {border: 1px solid rgba(99,153,34,0.4); border-radius: 12px; background: rgba(99,153,34,0.1); padding: 20px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;}
.verdict-PARTIALLY-TRUE {border: 1px solid rgba(245,158,11,0.4); border-radius: 12px; background: rgba(245,158,11,0.1); padding: 20px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;}
.verdict-OPINION {border: 1px solid rgba(139,92,246,0.4); border-radius: 12px; background: rgba(139,92,246,0.1); padding: 20px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;}
.verdict-UNCERTAIN {border: 1px solid rgba(156,163,175,0.4); border-radius: 12px; background: rgba(156,163,175,0.1); padding: 20px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;}

.verdict-dot-FALSE {width: 12px; height: 12px; border-radius: 50%; background: #E24B4A; margin-right: 16px; flex-shrink: 0; box-shadow: 0 0 10px rgba(226,75,74,0.5);}
.verdict-dot-MOSTLY-TRUE {width: 12px; height: 12px; border-radius: 50%; background: #639922; margin-right: 16px; flex-shrink: 0; box-shadow: 0 0 10px rgba(99,153,34,0.5);}
.verdict-dot-PARTIALLY-TRUE {width: 12px; height: 12px; border-radius: 50%; background: #F59E0B; margin-right: 16px; flex-shrink: 0; box-shadow: 0 0 10px rgba(245,158,11,0.5);}
.verdict-dot-OPINION {width: 12px; height: 12px; border-radius: 50%; background: #8B5CF6; margin-right: 16px; flex-shrink: 0; box-shadow: 0 0 10px rgba(139,92,246,0.5);}
.verdict-dot-UNCERTAIN {width: 12px; height: 12px; border-radius: 50%; background: #9CA3AF; margin-right: 16px; flex-shrink: 0; box-shadow: 0 0 10px rgba(156,163,175,0.5);}

.verdict-label {font-size: 18px; font-weight: 700; text-transform: uppercase;}
.verdict-label-FALSE {color: #E24B4A;}
.verdict-label-MOSTLY-TRUE {color: #639922;}
.verdict-label-PARTIALLY-TRUE {color: #F59E0B;}
.verdict-label-OPINION {color: #8B5CF6;}
.verdict-label-UNCERTAIN {color: #9CA3AF;}

.verdict-desc {font-size: 13px; opacity: 0.7; margin-top: 4px;}

.summary-text {font-size: 15px; line-height: 1.6; opacity: 0.9;}

.claim-card {border-bottom: 1px solid rgba(255,255,255,0.08); padding: 16px 0;}
.claim-card:last-child {border-bottom: none; padding-bottom: 0;}
.claim-card:first-child {padding-top: 0;}
.claim-header {display: flex; gap: 12px; align-items: flex-start; margin-bottom: 8px;}
.claim-text {font-size: 15px; font-weight: 500; flex-grow: 1; line-height: 1.5;}
.claim-badge-SUPPORTED {color: #639922; font-weight: bold; font-size: 11px; border: 1px solid rgba(99,153,34,0.5); background: rgba(99,153,34,0.1); padding: 4px 8px; border-radius: 6px; flex-shrink: 0; margin-top: 2px;}
.claim-badge-CONTRADICTED {color: #E24B4A; font-weight: bold; font-size: 11px; border: 1px solid rgba(226,75,74,0.5); background: rgba(226,75,74,0.1); padding: 4px 8px; border-radius: 6px; flex-shrink: 0; margin-top: 2px;}
.claim-badge-INSUFFICIENT_EVIDENCE {color: #9CA3AF; font-weight: bold; font-size: 11px; border: 1px solid rgba(156,163,175,0.5); background: rgba(156,163,175,0.1); padding: 4px 8px; border-radius: 6px; flex-shrink: 0; margin-top: 2px;}
.claim-reason {font-size: 14px; opacity: 0.75; line-height: 1.5; margin-bottom: 8px;}
.evidence-box {background: rgba(0,0,0,0.2); border-radius: 6px; padding: 10px; font-size: 12px; opacity: 0.6; line-height: 1.4; max-height: 100px; overflow-y: auto;}

.metric-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;}
.metric-box {background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 12px; text-align: center;}
.metric-label {font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.5; margin-bottom: 4px;}
.metric-val {font-size: 18px; font-weight: 600;}

.footer-note {font-size: 12px; opacity: 0.4; text-align: center; margin-top: 2.5rem; line-height: 1.6;}
.divider {border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0;}
</style>
""", unsafe_allow_html=True)

# ─── Graph Initialization ─────────────────────────────
@st.cache_resource
def get_graph():
    # Cache buster to ensure new nodes are loaded
    return build_graph()

app_graph = get_graph()

# ─── UI Layout ─────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-left">
    <div class="nav-dot"></div>
    <div>
      <div class="nav-title">Agentic News Credibility Analyzer</div>
      <div class="nav-sub">LangGraph RAG & Agentic Fact Verification</div>
    </div>
  </div>
  <span class="nav-badge">v5.0 Architecture</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="hint-text">Paste a news article to analyze its credibility using multi-step AI reasoning.</p>', unsafe_allow_html=True)

text_input = st.text_area(
    "Article text",
    height=200,
    placeholder="Paste a full news article or specific claim here...",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze = st.button("Run Analysis Pipeline", use_container_width=True)

if analyze:
    if not text_input.strip():
        st.warning("Please paste some text to analyze.")
    else:
        with st.spinner("Executing LangGraph Agent Workflow..."):
            # Ensure fresh state dictionary to prevent leakage between runs
            initial_state = {
                "input_text": text_input,
                "ml_prediction": -1,
                "ml_confidence": 0.0,
                "extracted_claims": [],
                "generated_queries": {},
                "retrieved_evidence": {},
                "cleaned_evidence": {},
                "scored_evidence": {},
                "ranked_evidence": {},
                "filtered_evidence": {},
                "verification_results": [],
                "credibility_score": None,
                "reasoning_summary": {}
            }
            try:
                result = app_graph.invoke(initial_state)
                
                # Extract Results
                verdict = result.get("credibility_score") or "UNCERTAIN"
                v_class = verdict.replace(" ", "-")
                
                ml_conf = result.get("ml_confidence", 0.0)
                ml_pred = result.get("ml_prediction", -1)
                ml_label = "FAKE" if ml_pred == 1 else "REAL" if ml_pred == 0 else "UNKNOWN"
                
                summary = result.get("reasoning_summary", {}).get("SUMMARY", "")
                analysis = result.get("reasoning_summary", {}).get("ANALYSIS", "")
                risks = result.get("reasoning_summary", {}).get("RISK_FACTORS", [])
                
                verifications = result.get("verification_results", [])
                
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                # 1. FINAL VERDICT
                st.markdown(f"""
                <div class="verdict-{v_class}">
                  <div style="display:flex;align-items:center;">
                    <div class="verdict-dot-{v_class}"></div>
                    <div>
                      <div class="verdict-label verdict-label-{v_class}">FINAL VERDICT: {verdict}</div>
                      <div class="verdict-desc">Aggregated via LLM reasoning and live evidence retrieval.</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 2. ML METRICS
                st.markdown(f"""
                <div class="metric-grid">
                  <div class="metric-box">
                    <div class="metric-label">Base ML Prediction</div>
                    <div class="metric-val" style="color: {'#E24B4A' if ml_pred == 1 else '#639922'};">{ml_label}</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-label">ML Confidence</div>
                    <div class="metric-val">{ml_conf:.1%}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 3. SUMMARY
                if summary or analysis:
                    st.markdown(f"""
                    <div class="section-card">
                      <div class="section-header"><span class="section-tag">Reasoning Summary</span></div>
                      <div class="section-body">
                        <p class="summary-text" style="font-weight: 600; margin-bottom: 8px;">{summary}</p>
                        <p class="summary-text">{analysis}</p>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # 4. RISKS
                if risks:
                    risks_html = "".join([f"<li style='font-size: 14px; opacity: 0.8; margin-bottom: 6px;'>{r}</li>" for r in risks])
                    st.markdown(f"""
                    <div class="section-card">
                      <div class="section-header"><span class="section-tag">Identified Risk Factors</span></div>
                      <div class="section-body"><ul style="margin: 0; padding-left: 20px;">{risks_html}</ul></div>
                    </div>
                    """, unsafe_allow_html=True)

                # 5. CLAIM-LEVEL ANALYSIS
                if verifications:
                    claims_html = ""
                    for v in verifications:
                        c_status = v.get("status", "INSUFFICIENT_EVIDENCE")
                        c_text = v.get("claim", "")
                        c_reason = v.get("reason", "")
                        
                        claims_html += f"""
                        <div class="claim-card">
                          <div class="claim-header">
                            <div class="claim-text">{c_text}</div>
                            <div class="claim-badge-{c_status}">{c_status}</div>
                          </div>
                          <div class="claim-reason">{c_reason}</div>
                        </div>
                        """
                        
                    with st.expander("🔍 View Detailed Claim Verifications", expanded=True):
                        st.markdown(f"""
                        <div class="section-card" style="margin-bottom: 0; border: none; background: transparent;">
                          <div class="section-body" style="padding: 0;">{claims_html}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # 6. DEBUGGING PANEL
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                with st.expander("🛠️ Debugging: Queries, Retrieval & Scoring"):
                    st.markdown("This panel shows the optimized search queries and how evidence was filtered based on semantic similarity.")
                    
                    filtered = result.get("filtered_evidence", {})
                    scored = result.get("scored_evidence", {})
                    generated_queries = result.get("generated_queries", {})
                    
                    for claim, scored_docs in scored.items():
                        st.markdown(f"**Claim:** *{claim}*")
                        
                        # Show generated queries
                        queries = generated_queries.get(claim, [])
                        if queries:
                            st.markdown("**Generated Search Queries:**")
                            for q in queries:
                                st.markdown(f"- `{q}`")
                                
                        filtered_docs = filtered.get(claim, [])
                        filtered_texts = [d["evidence"] for d in filtered_docs]
                        
                        st.markdown("**Retrieved Evidence Scoring:**")
                        if not scored_docs:
                            st.info("No evidence retrieved.")
                        else:
                            for doc in scored_docs:
                                sim = doc.get("similarity", 0.0)
                                ev = doc.get("evidence", "")
                                
                                # Status
                                if ev in filtered_texts:
                                    status_badge = "✅ **KEPT** (Passed threshold)"
                                else:
                                    status_badge = "❌ **DISCARDED** (Low relevance)"
                                    
                                st.markdown(f"- **Score:** `{sim:.2f}` | {status_badge}\\n  <div class='evidence-box'>{ev}</div>", unsafe_allow_html=True)
                        st.markdown("---")

                st.markdown("""
                <p class="footer-note">
                  Powered by a LangGraph Agent Architecture integrating ML, Tavily Search, and Groq LLMs.<br>
                  This is an AI-generated assessment. Always verify critical claims with official sources.
                </p>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error("Pipeline Execution Failed.")
                st.exception(e)
                st.info("Check your API keys in .streamlit/secrets.toml and ensure your models/ folder contains model.pkl and tfidf.pkl.")