import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'

const VERDICT_STYLES = {
  'MOSTLY-TRUE': { color: '#059669', bg: 'bg-[#ecfdf5]', border: 'border-[#a7f3d0]', label: 'Mostly True', icon: '✓' },
  'FALSE': { color: '#dc2626', bg: 'bg-[#fef2f2]', border: 'border-[#fecaca]', label: 'False', icon: '✕' },
  'PARTIALLY-TRUE': { color: '#d97706', bg: 'bg-[#fffbeb]', border: 'border-[#fde68a]', label: 'Partially True', icon: '⚠' },
  'OPINION': { color: '#7c3aed', bg: 'bg-[#f5f3ff]', border: 'border-[#ddd6fe]', label: 'Opinion', icon: '★' },
  'UNCERTAIN': { color: '#4b5563', bg: 'bg-[#f3f4f6]', border: 'border-[#e5e7eb]', label: 'Uncertain', icon: '?' },
}

export default function Analyzer() {
  const navigate = useNavigate()
  const location = useLocation()
  
  const [text, setText] = useState(location.state?.initialText || '')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (location.state?.initialText) {
      runAnalysis(location.state.initialText)
    }
  }, [])

  const runAnalysis = async (textToAnalyze) => {
    const payloadText = typeof textToAnalyze === 'string' ? textToAnalyze : text
    if (!payloadText.trim()) return
    
    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const apiUrl = `${import.meta.env.VITE_API_URL || 'http://localhost:5001'}/analyze`
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: payloadText.trim() }),
      })
      if (!res.ok) throw new Error('Request failed')
      const data = await res.json()
      setResult(data)
    } catch {
      setError('Pipeline failed. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const vs = result ? VERDICT_STYLES[result.verdict] || VERDICT_STYLES.UNCERTAIN : null

  return (
    <div className="min-h-screen bg-[#f8f9fa] font-sans selection:bg-[#4f46e5]/30 flex flex-col">
      
      {/* Top Navbar (Consistent with Landing) */}
      <nav className="bg-white flex items-center justify-between px-6 lg:px-12 py-4 border-b border-gray-100 sticky top-0 z-50">
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => navigate('/')}>
          <div className="w-7 h-7 bg-[#4f46e5] rounded flex items-center justify-center">
            <span className="text-white text-sm font-bold">V</span>
          </div>
          <span className="font-semibold text-gray-900 text-lg tracking-tight">VerifyAI <span className="text-xs text-gray-400 font-normal ml-1">News Credibility Asset Management</span></span>
        </div>
        
        <div className="hidden md:flex items-center gap-8 text-sm font-medium text-gray-600">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${loading ? 'bg-[#4f46e5] animate-pulse' : result ? 'bg-[#4f46e5]' : 'bg-gray-300'}`}></div>
            <span>{loading ? 'Analysis in progress...' : result ? 'Report Generated' : 'Ready for input'}</span>
          </div>
        </div>

        <button 
          onClick={() => navigate('/')}
          className="text-gray-500 hover:text-[#4f46e5] text-sm font-medium transition-colors"
        >
          ← Back to Home
        </button>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 relative w-full max-w-[1400px] mx-auto p-6 lg:p-12">
        
        {/* Empty State / Input State */}
        {!result && (
          <div className="relative min-h-[70vh] flex items-center justify-center">
            {/* Concentric Circles Background */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] border border-indigo-50 rounded-full"></div>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] border border-indigo-50 rounded-full"></div>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[1000px] h-[1000px] border border-indigo-50 rounded-full"></div>
            
            {/* Input Card */}
            <div className="relative z-10 w-full max-w-3xl bg-white p-10 md:p-14 rounded-2xl shadow-xl shadow-gray-200/50 border border-gray-100">
              <div className="inline-block bg-[#e0e7ff] text-[#4f46e5] px-4 py-1.5 rounded-sm text-xs font-bold uppercase tracking-widest mb-6 border border-[#c7d2fe]/50">
                Verification Desk
              </div>
              <h1 className="text-3xl md:text-4xl text-gray-800 font-light mb-8">
                Submit an article for <span className="font-semibold text-[#4f46e5]">analysis</span>
              </h1>
              
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste the full article, excerpt, or claim here..."
                className="w-full h-64 bg-gray-50 border border-gray-200 rounded-xl p-6 text-gray-700 text-lg leading-relaxed focus:outline-none focus:ring-2 focus:ring-[#4f46e5]/20 focus:border-[#4f46e5] transition-all resize-none mb-8"
              />
              
              {error && <div className="mb-6 p-4 bg-red-50 border border-red-100 text-red-600 rounded-lg text-sm">{error}</div>}
              
              <div className="flex justify-end">
                <button 
                  onClick={runAnalysis}
                  disabled={loading || !text.trim()}
                  className="bg-[#4f46e5] text-white px-10 py-3.5 rounded-sm text-sm font-medium hover:bg-[#4338ca] transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-[#4f46e5]/20 flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                      Running Agentic Pipeline...
                    </>
                  ) : (
                    <>Analyze Content <span>→</span></>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Results State */}
        {result && (
          <div className="animate-in fade-in duration-500">
            {/* Stepper / Breadcrumb */}
            <div className="w-full bg-white rounded-xl shadow-sm border border-gray-100 p-6 mb-8 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3 text-sm font-semibold text-gray-800">
                <span className="w-6 h-6 bg-[#4f46e5] text-white rounded-full flex items-center justify-center text-xs">✓</span>
                Pipeline Complete
              </div>
              <div className="hidden md:flex items-center gap-4 text-xs font-medium text-gray-400">
                <span className="text-[#4f46e5]">ML Baseline</span>
                <span>→</span>
                <span className="text-[#4f46e5]">Claim Extraction</span>
                <span>→</span>
                <span className="text-[#4f46e5]">Live Retrieval</span>
                <span>→</span>
                <span className="text-[#4f46e5]">Evidence Scoring</span>
                <span>→</span>
                <span className="text-[#4f46e5]">LLM Reasoning</span>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              
              {/* Left Column: Retained Input */}
              <div className="lg:col-span-5 bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden flex flex-col h-[800px]">
                <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
                  <div className="inline-block bg-gray-100 text-gray-600 px-3 py-1 rounded-sm text-[10px] font-bold uppercase tracking-widest">
                    Submitted Copy
                  </div>
                </div>
                <div className="p-8 overflow-y-auto flex-1 font-serif text-gray-700 leading-loose text-lg whitespace-pre-wrap">
                  {text}
                </div>
              </div>

              {/* Right Column: Report */}
              <div className="lg:col-span-7 flex flex-col gap-6">
                
                {/* Verdict Hero Card */}
                <div className={`${vs.bg} ${vs.border} border rounded-xl p-8 lg:p-10 flex flex-col md:flex-row items-center justify-between shadow-sm`}>
                  <div className="flex-1">
                    <div className="text-[10px] font-bold uppercase tracking-widest mb-3 opacity-70" style={{ color: vs.color }}>
                      Agentic Verdict
                    </div>
                    <div className="text-5xl md:text-6xl font-bold tracking-tight mb-4" style={{ color: vs.color }}>
                      {vs.label}
                    </div>
                    <p className="text-sm font-medium opacity-80" style={{ color: vs.color }}>
                      Aggregated via ML scoring and live semantic retrieval.
                    </p>
                  </div>
                  <div className="mt-6 md:mt-0 w-24 h-24 rounded-full border-4 flex items-center justify-center text-4xl font-bold" style={{ borderColor: vs.color, color: vs.color }}>
                    {vs.icon}
                  </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-3 gap-6">
                  <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
                    <div className="text-[10px] font-bold uppercase tracking-widest text-gray-400 mb-2">Initial Signal</div>
                    <div className={`text-2xl font-bold ${result.ml_label === 'REAL' ? 'text-[#059669]' : result.ml_label === 'FAKE' ? 'text-red-600' : 'text-gray-600'}`}>
                      {result.ml_label}
                    </div>
                  </div>
                  <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
                    <div className="text-[10px] font-bold uppercase tracking-widest text-gray-400 mb-2">Confidence</div>
                    <div className="text-2xl font-bold text-gray-900">
                      {(result.ml_confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
                    <div className="text-[10px] font-bold uppercase tracking-widest text-gray-400 mb-2">Claims Verified</div>
                    <div className="text-2xl font-bold text-gray-900">
                      {result.claims_count || 'N/A'}
                    </div>
                  </div>
                </div>

                {/* Assessment Body */}
                <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-8 flex-1">
                  <div className="inline-block bg-[#e0e7ff] text-[#4f46e5] px-3 py-1 rounded-sm text-[10px] font-bold uppercase tracking-widest mb-6 border border-[#c7d2fe]/50">
                    Detailed Assessment
                  </div>
                  
                  {result.summary && (
                    <p className="text-xl font-medium text-gray-900 leading-relaxed mb-6">
                      {result.summary}
                    </p>
                  )}
                  {result.analysis && (
                    <p className="text-base text-gray-600 leading-relaxed mb-10">
                      {result.analysis}
                    </p>
                  )}

                  {/* Risk Factors */}
                  {result.risks && result.risks.length > 0 && (
                    <div className="pt-8 border-t border-gray-100">
                      <h4 className="text-[10px] font-bold uppercase tracking-widest text-red-500 mb-6">Identified Risk Factors</h4>
                      <ul className="flex flex-col gap-4">
                        {result.risks.map((risk, i) => (
                          <li key={i} className="flex gap-4 items-start text-sm text-gray-700 bg-red-50/50 p-4 rounded-lg border border-red-50">
                            <span className="text-red-500 font-bold mt-0.5">!</span>
                            <span className="leading-relaxed">{risk}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

              </div>
            </div>
          </div>
        )}
      </main>

    </div>
  )
}
