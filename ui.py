import streamlit as st
import time
from agents import build_search_score_agent, build_search_reader_agent, writer_chain, critic_chain

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #e2ddd6;
}

.stApp {
    background: #080b10;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2.5rem 4rem; max-width: 1280px; }

/* ── Hero ── */
.hero {
    padding: 3rem 0 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2.5rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
    flex-wrap: wrap;
}
.hero-left {}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(94, 196, 164, 0.1);
    border: 1px solid rgba(94, 196, 164, 0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5ec4a4;
    margin-bottom: 1.2rem;
}
.hero-badge::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #5ec4a4;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.04em;
    color: #f0ece5;
    margin: 0 0 0.6rem;
}
.hero h1 em {
    font-style: normal;
    color: #5ec4a4;
}
.hero-desc {
    font-size: 0.95rem;
    font-weight: 400;
    color: #7a7570;
    max-width: 460px;
    line-height: 1.6;
    margin: 0;
}
.hero-stat {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.2rem;
}
.hero-stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 400;
    color: #5ec4a4;
    line-height: 1;
}
.hero-stat-label {
    font-size: 0.72rem;
    color: #4a4845;
    text-transform: uppercase;
    letter-spacing: 0.15em;
}

/* ── Input section ── */
.input-section {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 2rem;
}

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f0ece5 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.8rem 1.1rem !important;
    transition: all 0.2s !important;
    caret-color: #5ec4a4 !important;
}
.stTextInput > div > div > input::placeholder { color: #3f3d3a !important; }
.stTextInput > div > div > input:focus {
    border-color: rgba(94,196,164,0.45) !important;
    background: rgba(94,196,164,0.03) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(94,196,164,0.07) !important;
}
.stTextInput > label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #5a5854 !important;
    font-weight: 400 !important;
    margin-bottom: 0.5rem !important;
}

/* ── Button ── */
.stButton > button {
    background: #5ec4a4 !important;
    color: #080b10 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.72rem 1.8rem !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #72d4b5 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Chips ── */
.chips-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.chip-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #3a3836;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.73rem;
    color: #6e6b67;
    font-family: 'Space Grotesk', sans-serif;
    cursor: default;
}

/* ── Pipeline column header ── */
.col-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3a3836;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* ── Pipeline step ── */
.pipe-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1.1rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    transition: all 0.25s;
}
.pipe-step:last-child { border-bottom: none; }

.pipe-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    transition: all 0.25s;
}
.pipe-icon-waiting  { background: rgba(255,255,255,0.04); }
.pipe-icon-running  { background: rgba(94,196,164,0.12); }
.pipe-icon-done     { background: rgba(94,196,164,0.15); }

.pipe-body { flex: 1; min-width: 0; }
.pipe-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #c8c3bc;
    margin: 0 0 2px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.pipe-title-waiting { color: #4a4845; }
.pipe-title-running { color: #f0ece5; }
.pipe-title-done    { color: #c8c3bc; }

.pipe-sub {
    font-size: 0.73rem;
    color: #3a3836;
    font-family: 'Space Grotesk', sans-serif;
}
.pipe-sub-running { color: #5ec4a4; }
.pipe-sub-done    { color: #3a5a50; }

.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 4px;
    white-space: nowrap;
}
.badge-waiting { background: rgba(255,255,255,0.04); color: #3a3836; }
.badge-running { background: rgba(94,196,164,0.12); color: #5ec4a4; }
.badge-done    { background: rgba(94,196,164,0.1); color: #3a5a50; }

/* ── Progress bar ── */
.progress-track {
    height: 2px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    margin-top: 0.8rem;
    overflow: hidden;
}
.progress-bar {
    height: 100%;
    border-radius: 2px;
    background: #5ec4a4;
    transition: width 0.5s ease;
}

/* ── Results ── */
.results-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3a3836;
    margin: 2.5rem 0 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.results-count {
    font-size: 0.62rem;
    color: #5ec4a4;
}

.raw-panel {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.raw-panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a4845;
    margin-bottom: 0.8rem;
}
.raw-panel-body {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.75;
    color: #6a6560;
    white-space: pre-wrap;
    max-height: 300px;
    overflow-y: auto;
}

.report-wrap {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(94,196,164,0.15);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
}
.report-wrap-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5ec4a4;
    margin-bottom: 1.4rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(94,196,164,0.1);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.report-wrap-label::before {
    content: '';
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #5ec4a4;
}

.critic-wrap {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
}
.critic-wrap-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7a7570;
    margin-bottom: 1.4rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* ── Streamlit expander ── */
details { border: none !important; background: transparent !important; }
details summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #3a3836 !important;
    letter-spacing: 0.1em !important;
    cursor: pointer;
    padding: 0.6rem 0 !important;
}
details[open] summary { color: #5a5854 !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: transparent !important;
    color: #5ec4a4 !important;
    border: 1px solid rgba(94,196,164,0.3) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.15s !important;
    width: auto !important;
}
.stDownloadButton > button:hover {
    background: rgba(94,196,164,0.06) !important;
    border-color: rgba(94,196,164,0.5) !important;
}

.stSpinner > div { color: #5ec4a4 !important; }

/* ── Footer ── */
.footer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #282826;
    text-align: center;
    margin-top: 4rem;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


# ── Step renderer ─────────────────────────────────────────────────────────────
STEPS = [
    ("🔍", "Search Agent",  "Gathers recent web information"),
    ("📄", "Reader Agent",  "Scrapes & extracts deep content"),
    ("✍️", "Writer Chain",  "Drafts the full research report"),
    ("🔎", "Critic Chain",  "Reviews & scores the report"),
]
STEP_KEYS = ["search", "reader", "writer", "critic"]

def render_pipeline(results: dict, running: bool):
    st.markdown('<div class="col-header">Pipeline — 4 stages</div>', unsafe_allow_html=True)
    completed = list(results.keys())
    n_done = len(completed)

    for i, (icon, title, desc) in enumerate(STEPS):
        key = STEP_KEYS[i]
        if key in completed:
            state = "done"
        elif running:
            # first key not yet in results is currently running
            for k in STEP_KEYS:
                if k not in completed:
                    state = "running" if k == key else "waiting"
                    break
            else:
                state = "waiting"
        else:
            state = "waiting"

        badge_map = {
            "waiting": ("WAITING", "badge-waiting"),
            "running": ("● RUN", "badge-running"),
            "done":    ("✓ DONE", "badge-done"),
        }
        icon_cls_map = {
            "waiting": "pipe-icon-waiting",
            "running": "pipe-icon-running",
            "done":    "pipe-icon-done",
        }
        title_cls_map = {
            "waiting": "pipe-title-waiting",
            "running": "pipe-title-running",
            "done":    "pipe-title-done",
        }
        sub_cls_map = {
            "waiting": "",
            "running": "pipe-sub-running",
            "done":    "pipe-sub-done",
        }
        badge_text, badge_cls = badge_map[state]
        icon_cls    = icon_cls_map[state]
        title_cls   = title_cls_map[state]
        sub_cls     = sub_cls_map[state]

        st.markdown(f"""
        <div class="pipe-step">
            <div class="pipe-icon {icon_cls}">{icon}</div>
            <div class="pipe-body">
                <div class="pipe-title {title_cls}">
                    {title}
                    <span class="badge {badge_cls}">{badge_text}</span>
                </div>
                <div class="pipe-sub {sub_cls}">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Progress bar
    pct = int(n_done / len(STEPS) * 100)
    st.markdown(f"""
    <div class="progress-track">
        <div class="progress-bar" style="width:{pct}%"></div>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                color:#3a3836;margin-top:0.5rem;text-align:right;letter-spacing:0.1em;">
        {n_done}/{len(STEPS)} COMPLETE
    </div>
    """, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for key in ("results", "running", "done"):
    if key not in st.session_state:
        st.session_state[key] = {} if key == "results" else False


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <div class="hero-badge">Multi-Agent System · v1.0</div>
        <h1>Research<em>Mind</em></h1>
        <p class="hero-desc">
            Four specialized AI agents search, scrape, write, and critique —
            delivering a structured research report on any topic in minutes.
        </p>
    </div>
    <div class="hero-stat">
        <div class="hero-stat-num"></div>
        <div class="hero-stat-label">AI Agents</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_gap, col_right = st.columns([5, 0.4, 4])

with col_left:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Quantum computing breakthroughs in 2025",
        key="topic_input",
        label_visibility="visible",
    )
    run_btn = st.button("Run Research Pipeline →", use_container_width=True)
    st.markdown("""
    <div class="chips-row">
        <span class="chip-label">Try →</span>
        <span class="chip">LLM agents 2025</span>
        <span class="chip">CRISPR gene editing</span>
        <span class="chip">Fusion energy</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    render_pipeline(st.session_state.results, st.session_state.running)


# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.rerun()

if st.session_state.running and not st.session_state.done:
    results = {}
    topic_val = st.session_state.topic_input

    # Step 1 — Search
    with st.spinner("Search Agent gathering information…"):
        search_agent = build_search_score_agent()
        sr = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic_val}")]
        })
        results["search"] = sr["messages"][-1].content
        st.session_state.results = dict(results)

    # Step 2 — Reader
    with st.spinner("Reader Agent scraping top sources…"):
        reader_agent = build_search_reader_agent()
        rr = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic_val}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{results['search'][:800]}"
            )]
        })
        results["reader"] = rr["messages"][-1].content
        st.session_state.results = dict(results)

    # Step 3 — Writer
    with st.spinner("Writer drafting the report…"):
        research_combined = (
            f"SEARCH RESULTS:\n{results['search']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
        )
        results["writer"] = writer_chain.invoke({
            "topic": topic_val,
            "research": research_combined,
        })
        st.session_state.results = dict(results)

    # Step 4 — Critic
    with st.spinner("Critic reviewing the report…"):
        results["critic"] = critic_chain.invoke({
            "report": results["writer"]
        })
        st.session_state.results = dict(results)

    st.session_state.running = False
    st.session_state.done = True
    st.rerun()


# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r:
    n = len(r)
    st.markdown(f"""
    <div class="results-header">
        Output
        <span class="results-count">{n}/4 stages complete</span>
    </div>
    """, unsafe_allow_html=True)

    if "search" in r:
        with st.expander("🔍  Search results — raw", expanded=False):
            st.markdown(f"""
            <div class="raw-panel">
                <div class="raw-panel-label">Search Agent Output</div>
                <div class="raw-panel-body">{r['search']}</div>
            </div>
            """, unsafe_allow_html=True)

    if "reader" in r:
        with st.expander("📄  Scraped content — raw", expanded=False):
            st.markdown(f"""
            <div class="raw-panel">
                <div class="raw-panel-label">Reader Agent Output</div>
                <div class="raw-panel-body">{r['reader']}</div>
            </div>
            """, unsafe_allow_html=True)

    if "writer" in r:
        st.markdown("""
        <div class="report-wrap">
            <div class="report-wrap-label">Final Research Report</div>
        """, unsafe_allow_html=True)
        st.markdown(r["writer"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            label="Download report (.md)",
            data=r["writer"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    if "critic" in r:
        st.markdown("""
        <div class="critic-wrap">
            <div class="critic-wrap-label">Critic Feedback</div>
        """, unsafe_allow_html=True)
        st.markdown(r["critic"])
        st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ResearchMind · LangChain multi-agent pipeline · Streamlit
</div>
""", unsafe_allow_html=True)