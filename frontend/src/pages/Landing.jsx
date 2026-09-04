import { useNavigate } from 'react-router-dom'

// --- Reusable Components ---

const SectionBadge = ({ children }) => (
  <div className="inline-block bg-[#e0e7ff] text-[#4f46e5] px-4 py-1.5 rounded-sm text-[11px] font-bold uppercase tracking-widest mb-6 border border-[#c7d2fe]/50">
    {children}
  </div>
)

const PipelineStep = ({ number, title, description, isLast = false }) => (
  <div className="relative flex gap-6 items-start">
    {!isLast && (
      <div className="absolute left-6 top-14 bottom-[-40px] w-px bg-gradient-to-b from-[#4f46e5]/30 to-transparent"></div>
    )}
    <div className="relative z-10 shrink-0 w-12 h-12 rounded-full border border-[#4f46e5]/20 bg-[#e0e7ff] text-[#4f46e5] flex items-center justify-center text-sm font-bold shadow-sm">
      {number}
    </div>
    <div className="pb-10">
      <h3 className="text-xl font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600 leading-relaxed">{description}</p>
    </div>
  </div>
)

const TechCard = ({ title, highlight, description, specs }) => (
  <div className="bg-white border border-gray-100 rounded-xl p-8 lg:p-10 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.05)] hover:shadow-[0_8px_30px_-4px_rgba(79,70,229,0.1)] transition-all duration-300">
    <h3 className="text-2xl font-light text-gray-900 mb-4">
      {title} <span className="font-semibold text-[#4f46e5]">{highlight}</span>
    </h3>
    <p className="text-gray-600 leading-relaxed mb-8 h-20">
      {description}
    </p>
    <div className="pt-6 border-t border-gray-100 flex flex-wrap gap-2">
      {specs.map((spec, i) => (
        <span key={i} className="inline-block bg-gray-50 text-gray-500 border border-gray-200 px-3 py-1 rounded text-[10px] font-bold uppercase tracking-widest">
          {spec}
        </span>
      ))}
    </div>
  </div>
)

// --- Main Page ---

export default function Landing() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-white font-sans selection:bg-[#4f46e5]/30">
      
      {/* Top Navbar */}
      <nav className="bg-white/90 backdrop-blur-md flex items-center justify-between px-6 lg:px-12 py-4 border-b border-gray-100 sticky top-0 z-50">
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => window.scrollTo(0, 0)}>
          <div className="w-7 h-7 bg-[#4f46e5] rounded flex items-center justify-center shadow-sm">
            <span className="text-white text-sm font-bold">V</span>
          </div>
          <span className="font-semibold text-gray-900 text-lg tracking-tight">
            VerifyAI <span className="text-[11px] text-gray-400 font-normal ml-2 uppercase tracking-widest hidden sm:inline-block">News Credibility Asset Management</span>
          </span>
        </div>
        
        <div className="hidden md:flex items-center gap-8 text-sm font-medium text-gray-600">
          <a href="#about" className="hover:text-[#4f46e5] transition-colors">About</a>
          <a href="#pipeline" className="hover:text-[#4f46e5] transition-colors">Pipeline</a>
          <a href="#architecture" className="hover:text-[#4f46e5] transition-colors">Architecture</a>
        </div>

        <button 
          onClick={() => navigate('/analyze')}
          className="bg-[#4f46e5] text-white px-6 py-2.5 rounded text-sm font-medium hover:bg-[#4338ca] transition-colors shadow-sm shadow-[#4f46e5]/20"
        >
          Analyze News
        </button>
      </nav>

      {/* Hero Section */}
      <section className="relative h-[85vh] min-h-[600px] w-full flex flex-col justify-between overflow-hidden">
        <div className="absolute inset-0 bg-black/30 z-10"></div>
        <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent z-10"></div>
        {/* New Dark Global Network / Earth Image */}
        <img 
          src="https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2600&auto=format&fit=crop" 
          alt="Global digital network" 
          className="absolute inset-0 w-full h-full object-cover transform scale-105 motion-safe:animate-[pulse_20s_ease-in-out_infinite_alternate]"
        />

        <div className="relative z-20 px-8 lg:px-12 pt-28 max-w-[1400px] w-full mx-auto">
          {/* Updated Hero Text & Highlight Color */}
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight text-white leading-[1.05] max-w-4xl">
            Real-Time <span className="text-[#818cf8]">Truth Verification</span><br />
            for the Modern Web
          </h1>
          
          <button 
            onClick={() => navigate('/analyze')}
            className="mt-10 bg-[#4f46e5] text-white px-8 py-3.5 rounded-sm text-sm font-medium hover:bg-[#4338ca] transition-colors flex items-center gap-3 shadow-lg shadow-[#4f46e5]/25 group"
          >
            Access the Verification Desk <span className="transform group-hover:translate-x-1 transition-transform">→</span>
          </button>
        </div>

        <div className="relative z-20 px-8 lg:px-12 pb-12 flex flex-col lg:flex-row justify-between items-end gap-12 max-w-[1400px] w-full mx-auto">
          <div className="mb-4 lg:mb-0 w-full lg:w-auto">
            <p className="text-white/60 text-[10px] uppercase tracking-widest font-bold mb-4">Powered by modern infrastructure</p>
            <div className="flex flex-wrap items-center gap-x-10 gap-y-4 opacity-90">
              <div className="text-xl font-bold text-white flex items-center gap-2">
                <span className="text-indigo-400">❖</span> LangGraph
              </div>
              <div className="text-xl font-light tracking-widest text-white">
                TAVILY
              </div>
              <div className="text-lg font-bold text-white flex items-center gap-2">
                <span className="bg-white/90 text-black px-2 py-0.5 rounded text-sm">scikit</span> Learn
              </div>
              <div className="text-xl font-bold text-white flex items-center gap-1">
                groq
              </div>
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur-md border border-white/20 p-8 rounded-xl w-full lg:w-[420px] shadow-2xl">
            <p className="text-indigo-400 text-[10px] uppercase tracking-widest font-bold mb-4">Live Architecture · 5-Step Pipeline</p>
            <h3 className="text-white text-lg font-medium leading-relaxed mb-8">
              Understanding how our 99%-accurate ML baseline interacts dynamically with live semantic retrieval.
            </h3>
            <button 
              onClick={() => navigate('/analyze')}
              className="text-white text-sm font-medium flex items-center gap-2 hover:text-indigo-400 transition-colors group"
            >
              Test the pipeline <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
            </button>
          </div>
        </div>
      </section>

      {/* About Section (Concentric Circles) */}
      <section id="about" className="relative py-32 overflow-hidden bg-white flex flex-col items-center justify-center border-b border-gray-100">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[300px] h-[300px] border border-gray-100 rounded-full"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] border border-gray-100 rounded-full"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] border border-gray-100 rounded-full"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[1200px] h-[1200px] border border-gray-100 rounded-full"></div>

        <div className="relative z-10 text-center max-w-5xl px-6">
          <SectionBadge>About VerifyAI</SectionBadge>
          
          <h2 className="text-2xl md:text-[34px] leading-[1.7] text-gray-500 font-light text-center">
            an <span className="inline-flex items-center justify-center bg-white shadow-sm border border-gray-200 px-3 py-1 font-bold text-gray-900 rounded mx-1 text-[26px]">AGENTIC AI</span> pipeline via <span className="inline-flex items-center justify-center bg-white shadow-sm border border-gray-200 px-3 py-1 font-bold text-gray-900 rounded mx-1 text-[26px]">LangGraph</span> specialises in autonomous <span className="text-[#4f46e5] font-semibold">fact-checking</span> for large-scale <span className="text-[#4f46e5] font-semibold">Digital Media</span>, delivering fully verified credibility reports in <span className="text-gray-900 font-medium">under 8 seconds</span>.
          </h2>
        </div>
      </section>

      {/* Pipeline Section */}
      <section id="pipeline" className="py-24 bg-[#f8f9fa] border-b border-gray-100">
        <div className="max-w-[1400px] mx-auto px-6 lg:px-12 grid grid-cols-1 lg:grid-cols-2 gap-20 items-center">
          <div>
            <SectionBadge>Orchestration Workflow</SectionBadge>
            <h2 className="text-4xl lg:text-5xl font-light text-gray-900 mb-8 tracking-tight leading-tight">
              An <span className="font-semibold text-[#4f46e5]">agentic graph</span> that mimics human journalism.
            </h2>
            <p className="text-lg text-gray-600 mb-10 leading-relaxed">
              VerifyAI doesn't blindly trust a single model. It breaks complex articles down into atomic claims, searches the live web for contradictory evidence, scores sources by semantic relevance, and synthesizes a final verdict—all orchestrated autonomously by LangGraph.
            </p>
            <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100">
              <h4 className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-4">Routing Logic</h4>
              <p className="text-sm text-gray-700 leading-relaxed border-l-2 border-[#4f46e5] pl-4">
                If the initial ML baseline predicts a result with <strong className="text-gray-900">&gt;90% confidence</strong>, the system triggers a fast-path lightweight verification. Otherwise, it routes the text to the deep-research claim verification agents.
              </p>
            </div>
          </div>
          
          <div className="bg-white p-10 lg:p-14 rounded-2xl shadow-xl shadow-gray-200/40 border border-gray-100">
            <PipelineStep 
              number="01" 
              title="ML Baseline Prediction" 
              description="A TF-IDF and scikit-learn Logistic Regression classifier gives an initial REAL or FAKE signal with a strict confidence score."
            />
            <PipelineStep 
              number="02" 
              title="Claim Extraction" 
              description="The Groq LLM parses the raw article text into individual, verifiable factual claims, stripping out opinion and filler."
            />
            <PipelineStep 
              number="03" 
              title="Live Web Retrieval" 
              description="Optimized search queries are generated per claim and sent to Tavily to fetch real-time, authoritative sources."
            />
            <PipelineStep 
              number="04" 
              title="Evidence Scoring" 
              description="Retrieved documents are analyzed for positive/negative verification terms, discarding low-relevance noise."
            />
            <PipelineStep 
              number="05" 
              title="Final Synthesis" 
              description="Llama-3 synthesizes all signals into a final verdict (Mostly True, False, Opinion, etc.) with transparent risk factors."
              isLast={true}
            />
          </div>
        </div>
      </section>

      {/* Architecture Section */}
      <section id="architecture" className="py-24 bg-white">
        <div className="max-w-[1400px] mx-auto px-6 lg:px-12">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <SectionBadge>Dual-Engine Architecture</SectionBadge>
            <h2 className="text-4xl lg:text-5xl font-light text-gray-900 mb-6 tracking-tight">
              Speed of ML. <span className="font-semibold text-[#4f46e5]">Depth of Generative AI.</span>
            </h2>
            <p className="text-lg text-gray-600 leading-relaxed">
              By combining traditional machine learning classifiers with modern LLM reasoning and live RAG (Retrieval-Augmented Generation), we eliminate hallucinations and ensure maximum trust.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <TechCard 
              title="The Baseline"
              highlight="Classifier"
              description="Trained on the ISOT Fake News dataset containing over 44,000 verified authentic and fake news articles. It acts as the ultra-fast first line of defense."
              specs={['TF-IDF Vectorization', 'Logistic Regression', '99% Historical Accuracy', 'Sub-second Latency']}
            />
            <TechCard 
              title="The Agentic"
              highlight="Fallback"
              description="When confidence is low, the pipeline falls back to an advanced reasoning engine that reads the live internet, grounding every claim in current reality."
              specs={['Groq Llama 3.1 8B', 'Tavily Advanced Search', 'Semantic Scoring', 'Zero Hallucinations']}
            />
          </div>
        </div>
      </section>

      {/* Bottom CTA */}
      <section className="bg-[#090a0c] py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-[#4f46e5]/20 via-transparent to-transparent"></div>
        <div className="max-w-4xl mx-auto px-6 text-center relative z-10">
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6 tracking-tight">
            Stop trusting blindly.<br />Start <span className="text-[#818cf8]">verifying</span>.
          </h2>
          <p className="text-lg text-white/60 mb-10 max-w-2xl mx-auto">
            Paste any news article, excerpt, or social media claim and get a full credibility report powered by agentic AI in seconds.
          </p>
          <button 
            onClick={() => navigate('/analyze')}
            className="bg-[#4f46e5] text-white px-10 py-4 rounded-sm text-sm font-bold tracking-wide hover:bg-[#4338ca] transition-all shadow-xl shadow-[#4f46e5]/20"
          >
            Access the Verification Desk
          </button>
        </div>
      </section>

      {/* Simple Footer */}
      <footer className="bg-white border-t border-gray-100 py-8 text-center text-sm text-gray-400">
        <p>© 2026 VerifyAI. Built with React, Tailwind, Flask, and LangGraph.</p>
      </footer>

    </div>
  )
}
