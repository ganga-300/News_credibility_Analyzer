import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

const VERDICT_STYLES = {
  'MOSTLY-TRUE': { color: '#639922', bg: 'rgba(99,153,34,0.08)', border: 'rgba(99,153,34,0.25)' },
  FALSE: { color: '#E24B4A', bg: 'rgba(226,75,74,0.08)', border: 'rgba(226,75,74,0.25)' },
  'PARTIALLY-TRUE': { color: '#F59E0B', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.25)' },
  OPINION: { color: '#8B5CF6', bg: 'rgba(139,92,246,0.08)', border: 'rgba(139,92,246,0.25)' },
  UNCERTAIN: { color: '#9CA3AF', bg: 'rgba(156,163,175,0.08)', border: 'rgba(156,163,175,0.25)' },
}

const CLAIM_BADGE = {
  SUPPORTED: { color: '#639922', bg: 'rgba(99,153,34,0.12)', border: 'rgba(99,153,34,0.3)' },
  CONTRADICTED: { color: '#E24B4A', bg: 'rgba(226,75,74,0.12)', border: 'rgba(226,75,74,0.3)' },
  INSUFFICIENT_EVIDENCE: { color: '#9CA3AF', bg: 'rgba(156,163,175,0.12)', border: 'rgba(156,163,175,0.3)' },
}

export default function Analyzer() {
  const navigate = useNavigate()
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const runAnalysis = async () => {
    if (!text.trim()) return
    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const res = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      })
      if (!res.ok) throw new Error('Request failed')
      const data = await res.json()
      setResult(data)
    } catch {
      setError('Pipeline failed. Make sure the backend is running on port 5000.')
    } finally {
      setLoading(false)
    }
  }

  const vs = result ? VERDICT_STYLES[result.verdict] || VERDICT_STYLES.UNCERTAIN : null

  return (
    <div style={{ minHeight: '100vh', background: '#0a0c10' }}>
      {/* ─── Navbar ─── */}
      <nav
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 40px',
          height: '52px',
          borderBottom: '0.5px solid rgba(255,255,255,0.08)',
        }}
      >
        <button
          onClick={() => navigate('/')}
          style={{
            background: 'none',
            border: 'none',
            color: 'rgba(232,234,240,0.5)',
            fontSize: '13px',
            cursor: 'pointer',
            transition: 'color 0.2s',
          }}
          onMouseEnter={(e) => (e.target.style.color = '#e8eaf0')}
          onMouseLeave={(e) => (e.target.style.color = 'rgba(232,234,240,0.5)')}
        >
          ← Back
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div
            style={{
              width: '7px',
              height: '7px',
              borderRadius: '50%',
              background: '#378ADD',
            }}
          />
          <span style={{ color: '#fff', fontSize: '14px', fontWeight: 500 }}>
            VerifyAI
          </span>
        </div>

        <div
          style={{
            background: 'rgba(55,138,221,0.08)',
            border: '0.5px solid rgba(55,138,221,0.25)',
            borderRadius: '20px',
            padding: '4px 12px',
            fontSize: '11px',
            color: '#378ADD',
          }}
        >
          v5.0 · LangGraph
        </div>
      </nav>

      {/* ─── Main content ─── */}
      <div style={{ maxWidth: '680px', margin: '0 auto', padding: '40px 24px 64px' }}>
        {/* ─── Input ─── */}
        <div style={{ marginBottom: '32px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '11px',
              fontWeight: 600,
              letterSpacing: '1.5px',
              color: 'rgba(232,234,240,0.5)',
              textTransform: 'uppercase',
              marginBottom: '10px',
            }}
          >
            PASTE ARTICLE
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste the full article text here..."
            style={{
              width: '100%',
              minHeight: '140px',
              background: 'rgba(255,255,255,0.03)',
              border: '0.5px solid rgba(255,255,255,0.1)',
              borderRadius: '10px',
              padding: '16px',
              fontSize: '13px',
              color: '#e8eaf0',
              resize: 'vertical',
              lineHeight: 1.7,
              fontFamily: 'inherit',
            }}
          />
          <div
            style={{
              fontSize: '11px',
              color: 'rgba(232,234,240,0.3)',
              marginTop: '8px',
            }}
          >
            Supports full articles, excerpts, or single claims
          </div>
        </div>

        <button
          onClick={runAnalysis}
          disabled={loading || !text.trim()}
          style={{
            width: '100%',
            background: loading ? 'rgba(55,138,221,0.5)' : '#378ADD',
            color: '#fff',
            border: 'none',
            padding: '12px',
            borderRadius: '8px',
            fontSize: '14px',
            fontWeight: 500,
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'opacity 0.2s',
            marginBottom: '36px',
            opacity: !text.trim() ? 0.4 : 1,
          }}
          onMouseEnter={(e) => {
            if (!loading && text.trim()) e.target.style.opacity = '0.85'
          }}
          onMouseLeave={(e) => {
            if (!loading && text.trim()) e.target.style.opacity = '1'
          }}
        >
          {loading ? 'Analyzing...' : '▶ Run analysis pipeline'}
        </button>

        {/* ─── Error ─── */}
        {error && (
          <div
            className="fade-in"
            style={{
              background: 'rgba(226,75,74,0.08)',
              border: '0.5px solid rgba(226,75,74,0.25)',
              borderRadius: '10px',
              padding: '20px 24px',
              color: '#E24B4A',
              fontSize: '13px',
              marginBottom: '24px',
            }}
          >
            {error}
          </div>
        )}

        {/* ─── Results ─── */}
        {result && (
          <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {/* Verdict bar */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: vs.bg,
                border: `0.5px solid ${vs.border}`,
                borderRadius: '10px',
                padding: '20px 24px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div
                  style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    background: vs.color,
                  }}
                />
                <div>
                  <div style={{ fontSize: '15px', fontWeight: 600, color: vs.color }}>
                    {result.verdict}
                  </div>
                  <div style={{ fontSize: '11px', color: 'rgba(232,234,240,0.5)', marginTop: '2px' }}>
                    Aggregated via LLM reasoning · live evidence retrieval
                  </div>
                </div>
              </div>
              <div style={{ fontSize: '28px', fontWeight: 600, color: vs.color }}>
                {Math.round(result.ml_confidence * 100)}%
              </div>
            </div>

            {/* Metrics row */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
              {[
                {
                  label: 'ML Prediction',
                  value: result.ml_label,
                  color: result.ml_label === 'REAL' ? '#639922' : result.ml_label === 'FAKE' ? '#E24B4A' : '#9CA3AF',
                },
                {
                  label: 'ML Confidence',
                  value: `${(result.ml_confidence * 100).toFixed(1)}%`,
                  color: '#378ADD',
                },
                {
                  label: 'Claims verified',
                  value: result.claims_count || 0,
                  color: '#e8eaf0',
                },
              ].map((m, i) => (
                <div
                  key={i}
                  style={{
                    background: 'rgba(255,255,255,0.03)',
                    border: '0.5px solid rgba(255,255,255,0.08)',
                    borderRadius: '10px',
                    padding: '20px',
                    textAlign: 'center',
                  }}
                >
                  <div style={{ fontSize: '11px', color: 'rgba(232,234,240,0.5)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 600 }}>
                    {m.label}
                  </div>
                  <div style={{ fontSize: '20px', fontWeight: 600, color: m.color }}>
                    {m.value}
                  </div>
                </div>
              ))}
            </div>

            {/* Reasoning summary */}
            {(result.summary || result.analysis) && (
              <div
                style={{
                  background: 'rgba(255,255,255,0.03)',
                  border: '0.5px solid rgba(255,255,255,0.08)',
                  borderRadius: '10px',
                  padding: '24px',
                }}
              >
                <div
                  style={{
                    fontSize: '10px',
                    fontWeight: 600,
                    letterSpacing: '1.5px',
                    color: 'rgba(232,234,240,0.5)',
                    textTransform: 'uppercase',
                    marginBottom: '14px',
                  }}
                >
                  REASONING SUMMARY
                </div>
                {result.summary && (
                  <p style={{ fontSize: '13px', fontWeight: 500, color: '#e8eaf0', lineHeight: 1.7, marginBottom: result.analysis ? '10px' : 0 }}>
                    {result.summary}
                  </p>
                )}
                {result.analysis && (
                  <p style={{ fontSize: '13px', color: 'rgba(232,234,240,0.5)', lineHeight: 1.7 }}>
                    {result.analysis}
                  </p>
                )}
              </div>
            )}

            {/* Risk factors */}
            {result.risks && result.risks.length > 0 && (
              <div
                style={{
                  background: 'rgba(255,255,255,0.03)',
                  border: '0.5px solid rgba(255,255,255,0.08)',
                  borderRadius: '10px',
                  padding: '24px',
                }}
              >
                <div
                  style={{
                    fontSize: '10px',
                    fontWeight: 600,
                    letterSpacing: '1.5px',
                    color: 'rgba(232,234,240,0.5)',
                    textTransform: 'uppercase',
                    marginBottom: '14px',
                  }}
                >
                  RISK FACTORS
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {result.risks.map((risk, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                      <div
                        style={{
                          width: '5px',
                          height: '5px',
                          borderRadius: '50%',
                          background: '#F59E0B',
                          marginTop: '6px',
                          flexShrink: 0,
                        }}
                      />
                      <span style={{ fontSize: '12px', color: 'rgba(232,234,240,0.5)', lineHeight: 1.6 }}>
                        {risk}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Claim verifications */}
            {result.verification_results && result.verification_results.length > 0 && (
              <div
                style={{
                  background: 'rgba(255,255,255,0.03)',
                  border: '0.5px solid rgba(255,255,255,0.08)',
                  borderRadius: '10px',
                  padding: '24px',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '14px',
                  }}
                >
                  <div
                    style={{
                      fontSize: '10px',
                      fontWeight: 600,
                      letterSpacing: '1.5px',
                      color: 'rgba(232,234,240,0.5)',
                      textTransform: 'uppercase',
                    }}
                  >
                    CLAIM VERIFICATIONS
                  </div>
                  <div style={{ fontSize: '11px', color: 'rgba(232,234,240,0.3)' }}>
                    {result.verification_results.length} claims extracted
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  {result.verification_results.map((claim, i) => {
                    const badge = CLAIM_BADGE[claim.status] || CLAIM_BADGE.INSUFFICIENT_EVIDENCE
                    return (
                      <div
                        key={i}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: '14px 0',
                          borderTop: i > 0 ? '0.5px solid rgba(255,255,255,0.06)' : 'none',
                        }}
                      >
                        <span style={{ fontSize: '13px', color: 'rgba(232,234,240,0.75)', flex: 1, marginRight: '16px' }}>
                          {claim.text}
                        </span>
                        <span
                          style={{
                            fontSize: '10px',
                            fontWeight: 600,
                            letterSpacing: '0.5px',
                            padding: '3px 10px',
                            borderRadius: '4px',
                            background: badge.bg,
                            border: `0.5px solid ${badge.border}`,
                            color: badge.color,
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {claim.status}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
