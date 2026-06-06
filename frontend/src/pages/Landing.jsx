import { useNavigate } from 'react-router-dom'

const features = [
  {
    icon: '⬡',
    title: 'Agentic pipeline',
    desc: 'LangGraph orchestrates claim extraction, query generation, retrieval, scoring, and reasoning automatically.',
  },
  {
    icon: '◎',
    title: 'Live evidence retrieval',
    desc: 'Tavily fetches real-time sources and scores them by semantic similarity to filter noise.',
  },
  {
    icon: '⊕',
    title: 'ML + LLM fusion',
    desc: 'ISOT-trained classifier combined with Groq Llama reasoning for a reliable, explainable verdict.',
  },
]

const steps = [
  {
    num: 1,
    title: 'ML baseline prediction',
    desc: 'TF-IDF + scikit-learn classifier gives an initial REAL/FAKE signal with confidence score.',
  },
  {
    num: 2,
    title: 'Claim extraction',
    desc: 'Groq LLM parses the article into individual verifiable factual claims.',
  },
  {
    num: 3,
    title: 'Query generation & retrieval',
    desc: 'Optimized search queries generated per claim and sent to Tavily for live evidence.',
  },
  {
    num: 4,
    title: 'Evidence scoring & filtering',
    desc: 'Retrieved documents ranked by semantic similarity, low-relevance evidence discarded.',
  },
  {
    num: 5,
    title: 'Final verdict & reasoning',
    desc: 'LLM synthesizes all signals into a final verdict: Mostly True, False, Partially True, Opinion, or Uncertain.',
  },
]

export default function Landing() {
  const navigate = useNavigate()

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

        <div style={{ display: 'flex', gap: '32px' }}>
          {['How it works', 'Tech stack', 'About'].map((item) => (
            <a
              key={item}
              href={`#${item.toLowerCase().replace(/\s/g, '-')}`}
              style={{
                color: 'rgba(232,234,240,0.5)',
                fontSize: '13px',
                textDecoration: 'none',
                transition: 'color 0.2s',
              }}
              onMouseEnter={(e) => (e.target.style.color = '#e8eaf0')}
              onMouseLeave={(e) =>
                (e.target.style.color = 'rgba(232,234,240,0.5)')
              }
            >
              {item}
            </a>
          ))}
        </div>

        <button
          onClick={() => navigate('/analyze')}
          style={{
            background: '#378ADD',
            color: '#fff',
            border: 'none',
            padding: '6px 16px',
            borderRadius: '6px',
            fontSize: '13px',
            fontWeight: 500,
            cursor: 'pointer',
            transition: 'opacity 0.2s',
          }}
          onMouseEnter={(e) => (e.target.style.opacity = '0.85')}
          onMouseLeave={(e) => (e.target.style.opacity = '1')}
        >
          Try it free →
        </button>
      </nav>

      {/* ─── Hero ─── */}
      <section
        style={{
          maxWidth: '680px',
          margin: '0 auto',
          textAlign: 'center',
          padding: '80px 24px 0',
        }}
      >
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            background: 'rgba(55,138,221,0.08)',
            border: '0.5px solid rgba(55,138,221,0.25)',
            borderRadius: '20px',
            padding: '5px 14px',
            fontSize: '12px',
            color: '#378ADD',
            marginBottom: '28px',
          }}
        >
          ✦ LangGraph · Groq · Tavily
        </div>

        <h1
          style={{
            fontSize: '38px',
            fontWeight: 500,
            lineHeight: 1.25,
            color: '#e8eaf0',
            marginBottom: '18px',
          }}
        >
          Don't trust news.
          <br />
          <span style={{ color: '#378ADD' }}>Verify it.</span>
        </h1>

        <p
          style={{
            fontSize: '15px',
            color: 'rgba(232,234,240,0.5)',
            lineHeight: 1.7,
            maxWidth: '520px',
            margin: '0 auto 32px',
          }}
        >
          Paste any article and our agentic AI pipeline extracts claims,
          retrieves live evidence, and delivers a credibility verdict in seconds.
        </p>

        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '12px',
            flexWrap: 'wrap',
          }}
        >
          <button
            onClick={() => navigate('/analyze')}
            style={{
              background: '#378ADD',
              color: '#fff',
              border: 'none',
              padding: '10px 24px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'opacity 0.2s',
            }}
            onMouseEnter={(e) => (e.target.style.opacity = '0.85')}
            onMouseLeave={(e) => (e.target.style.opacity = '1')}
          >
            Analyze an article →
          </button>
          <a
            href="#how-it-works"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              color: 'rgba(232,234,240,0.6)',
              border: '0.5px solid rgba(255,255,255,0.12)',
              padding: '10px 24px',
              borderRadius: '8px',
              fontSize: '14px',
              textDecoration: 'none',
              transition: 'border-color 0.2s',
              cursor: 'pointer',
            }}
            onMouseEnter={(e) =>
              (e.target.style.borderColor = 'rgba(255,255,255,0.25)')
            }
            onMouseLeave={(e) =>
              (e.target.style.borderColor = 'rgba(255,255,255,0.12)')
            }
          >
            See how it works
          </a>
        </div>
      </section>

      {/* ─── Stats bar ─── */}
      <section
        style={{
          maxWidth: '680px',
          margin: '64px auto 0',
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          borderTop: '0.5px solid rgba(255,255,255,0.07)',
          borderBottom: '0.5px solid rgba(255,255,255,0.07)',
        }}
      >
        {[
          { val: '44K+', label: 'Articles trained on' },
          { val: '5-step', label: 'Agentic pipeline' },
          { val: '~8s', label: 'Avg analysis time' },
        ].map((s, i) => (
          <div
            key={i}
            style={{
              textAlign: 'center',
              padding: '28px 0',
              borderLeft:
                i > 0 ? '0.5px solid rgba(255,255,255,0.07)' : 'none',
            }}
          >
            <div
              style={{ fontSize: '26px', fontWeight: 600, color: '#378ADD' }}
            >
              {s.val}
            </div>
            <div
              style={{
                fontSize: '12px',
                color: 'rgba(232,234,240,0.5)',
                marginTop: '4px',
              }}
            >
              {s.label}
            </div>
          </div>
        ))}
      </section>

      {/* ─── Features ─── */}
      <section
        id="tech-stack"
        style={{
          maxWidth: '820px',
          margin: '72px auto 0',
          padding: '0 24px',
        }}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 1fr',
            border: '0.5px solid rgba(255,255,255,0.07)',
            borderRadius: '10px',
            overflow: 'hidden',
          }}
        >
          {features.map((f, i) => (
            <div
              key={i}
              style={{
                padding: '32px 24px',
                borderLeft:
                  i > 0 ? '0.5px solid rgba(255,255,255,0.07)' : 'none',
              }}
            >
              <div
                style={{
                  width: '34px',
                  height: '34px',
                  borderRadius: '8px',
                  background: 'rgba(55,138,221,0.1)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '16px',
                  color: '#378ADD',
                  marginBottom: '16px',
                }}
              >
                {f.icon}
              </div>
              <div
                style={{
                  fontSize: '13px',
                  fontWeight: 500,
                  color: '#e8eaf0',
                  marginBottom: '6px',
                }}
              >
                {f.title}
              </div>
              <div
                style={{
                  fontSize: '12px',
                  color: 'rgba(232,234,240,0.5)',
                  lineHeight: 1.6,
                }}
              >
                {f.desc}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ─── How it works ─── */}
      <section
        id="how-it-works"
        style={{
          maxWidth: '620px',
          margin: '80px auto 0',
          padding: '0 24px',
        }}
      >
        <div
          style={{
            fontSize: '11px',
            fontWeight: 600,
            letterSpacing: '1.5px',
            color: '#378ADD',
            textTransform: 'uppercase',
            marginBottom: '10px',
          }}
        >
          HOW IT WORKS
        </div>
        <h2
          style={{
            fontSize: '20px',
            fontWeight: 500,
            color: '#e8eaf0',
            marginBottom: '36px',
          }}
        >
          Five-step verification pipeline
        </h2>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
          {steps.map((s, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                gap: '16px',
                padding: '20px 0',
                borderTop:
                  i > 0 ? '0.5px solid rgba(255,255,255,0.07)' : 'none',
              }}
            >
              <div
                style={{
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  background: 'rgba(55,138,221,0.12)',
                  color: '#378ADD',
                  fontSize: '12px',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                {s.num}
              </div>
              <div>
                <div
                  style={{
                    fontSize: '13px',
                    fontWeight: 500,
                    color: '#e8eaf0',
                    marginBottom: '4px',
                  }}
                >
                  {s.title}
                </div>
                <div
                  style={{
                    fontSize: '12px',
                    color: 'rgba(232,234,240,0.5)',
                    lineHeight: 1.6,
                  }}
                >
                  {s.desc}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ─── Footer CTA ─── */}
      <section
        style={{
          textAlign: 'center',
          padding: '80px 24px 64px',
          maxWidth: '520px',
          margin: '0 auto',
        }}
      >
        <h2
          style={{
            fontSize: '22px',
            fontWeight: 500,
            color: '#e8eaf0',
            marginBottom: '10px',
          }}
        >
          Ready to fact-check?
        </h2>
        <p
          style={{
            fontSize: '14px',
            color: 'rgba(232,234,240,0.5)',
            marginBottom: '28px',
          }}
        >
          Paste any news article and get a full credibility report.
        </p>
        <button
          onClick={() => navigate('/analyze')}
          style={{
            background: '#378ADD',
            color: '#fff',
            border: 'none',
            padding: '10px 28px',
            borderRadius: '8px',
            fontSize: '14px',
            fontWeight: 500,
            cursor: 'pointer',
            transition: 'opacity 0.2s',
          }}
          onMouseEnter={(e) => (e.target.style.opacity = '0.85')}
          onMouseLeave={(e) => (e.target.style.opacity = '1')}
        >
          Start analyzing →
        </button>
      </section>
    </div>
  )
}
