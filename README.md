<div align="center">

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=700&size=40&duration=3000&pause=1000&color=5EC4A4&center=true&vCenter=true&width=600&lines=ResearchMind+%F0%9F%94%AC;Multi-Agent+Research+AI;Search+%E2%86%92+Read+%E2%86%92+Write+%E2%86%92+Critique" alt="Typing SVG" />

<br/><br/>

<p align="center">
  <a href="https://multi-agents-research-system.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Try%20it%20Now-5EC4A4?style=for-the-badge&logoColor=white" alt="Live Demo"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  &nbsp;
  <img src="https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white" alt="LangChain"/>
  &nbsp;
 <img src="https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white" alt="LangChain"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Groq-LLaMA%203.3-F55036?style=for-the-badge&logoColor=white" alt="Groq"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Tavily-Search%20API-5EC4A4?style=for-the-badge" alt="Tavily"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Corrective%20RAG-blueviolet?style=flat-square" alt="Corrective RAG"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Agents-4%20Specialized-orange?style=flat-square" alt="Agents"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Status-Live%20%26%20Running-brightgreen?style=flat-square" alt="Status"/>
  &nbsp;
</p>

<br/>

> **A production-grade multi-agent AI system that autonomously searches the web, scrapes deep content, writes structured research reports, and self-critiques them - built from scratch with LangChain, Groq, and Tavily.**

<br/>

</div>

---

## 📌 Table of Contents

- [✨ What is ResearchMind?](#-what-is-researchmind)
- [🧠 The Architecture - Corrective RAG](#-the-architecture--corrective-rag)
- [🔬 Research Paper Inspiration](#-research-paper-inspiration)
- [⚙️ Agent Pipeline - Deep Dive](#️-agent-pipeline--deep-dive)
- [🗂️ Project Structure](#️-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🌐 Live Demo](#-live-demo)
- [🛠️ Tech Stack](#️-tech-stack)
- [📈 What I Learned](#-what-i-learned)
- [🔮 Future Roadmap](#-future-roadmap)
- [🙌 Acknowledgements](#-acknowledgements)

---

## ✨ What is ResearchMind?

**ResearchMind** is an autonomous, multi-agent research assistant powered by an advanced **Corrective RAG (CRAG)** architecture. You give it *any topic*, and four specialized AI agents collaborate in a pipeline to:

1. 🔍 **Search** the web for fresh, reliable information  
2. 📄 **Scrape** and extract deep content from the most relevant source  
3. ✍️ **Write** a structured, professional research report  
4. 🔎 **Critique** the report with scores, strengths, and improvements  

All of this happens autonomously - no human-in-the-loop, no copy-pasting, no tab-switching. Just one topic in, one polished report out.

<div align="center">

```
User Input ──▶  Search Agent ──▶  Reader Agent ──▶  Writer Chain ──▶  Critic Chain ──▶  Report + Feedback
                   │                    │                  │                 │
               Tavily API          BeautifulSoup       LLaMA 3.3         LLaMA 3.3
               (Web Search)        (HTML Scraper)      (Groq LLM)        (Groq LLM)
```

</div>

---

## 🧠 The Architecture - Corrective RAG

This project implements **Corrective RAG (CRAG)** - an advanced variant of Retrieval-Augmented Generation that goes beyond simple "retrieve + generate."

### Why Corrective RAG?

Standard RAG retrieves documents and feeds them to an LLM. The problem? Retrieved content is often noisy, irrelevant, or shallow. **CRAG fixes this** by adding a *correction* and *evaluation* loop.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CORRECTIVE RAG LOOP                          │
│                                                                 │
│   Query ──▶ Retrieve ──▶ Evaluate Quality                       │
│                                ├── CORRECT ──▶ Web Search       │
│                                │      └──▶ Re-rank & Filter     │
│                                └── GOOD ──▶ Generate Response   │
│                                                                 │
│   + Self-Critique Layer (Critic Agent evaluates final output)   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Differences from Vanilla RAG

| Feature | Standard RAG | **ResearchMind (CRAG)** |
|--------|-------------|-------------------------|
| Retrieval Source | Static vector DB | 🌐 Live web search |
| Content Quality | No validation | ✅ Agent-evaluated |
| Scraping | Not included | 📄 Deep HTML extraction |
| Self-Correction | ❌ None | ✔️ Critic Agent scores output |
| Output Format | Raw text | 📝 Structured markdown report |
| Feedback Loop | ❌ None | 🔁 Score + actionable critique |

---

## 🔬 Research Paper Inspiration

This project is grounded in real academic research on advanced RAG systems:

**📄 Corrective Retrieval Augmented Generation** - Yan et al., 2024  
The original CRAG paper proposes a retrieval evaluator that assesses the quality of retrieved documents and triggers corrective actions (web search, document refinement) when retrieval quality is low. This system implements the **core CRAG spirit**: autonomous evaluation + web-augmented correction.

**Key insights from the paper applied here:**
- Retrieved content must be *assessed*, not blindly used
- Web search as a fallback ensures up-to-date, high-quality grounding
- Decompose-then-recompose strategies improve generation quality
- Self-reflection via a separate critique step reduces hallucination

---

## ⚙️ Agent Pipeline - Deep Dive

### Stage 1 - 🔍 Search Agent

```python
def build_search_score_agent():
    return create_agent(model=llm, tools=[web_search])
```

- Uses **Tavily Search API** to find recent, high-quality web results
- Returns **titles, URLs, and content snippets** for top 5 results
- Acts as the *grounding layer* - ensures all information is real and current
- Model: **LLaMA 3.3 70B** via Groq (ultra-fast inference)

### Stage 2 - 📄 Reader Agent

```python
def build_search_reader_agent():
    return create_agent(model=llm, tools=[scrape_url])
```

- Reads Stage 1 output and **intelligently picks the most relevant URL**
- Scrapes the full page using `BeautifulSoup` - strips noise (nav, footer, scripts)
- Returns up to **3,000 characters** of clean, readable body content
- This is the *depth layer* - adds substance beyond snippets

### Stage 3 - ✍️ Writer Chain

```python
writer_chain = writer_prompt | llm | StrOutputParser()
```

- Combines search snippets + scraped content into a single rich context
- Generates a **structured research report** with:
  - Introduction
  - Key Findings (minimum 3 points)
  - Conclusion
  - Sources
- Output is **downloadable as `.md`**

### Stage 4 - 🔎 Critic Chain *(The CRAG Touch)*

```python
critic_chain = critic_prompt | llm | StrOutputParser()
```

- Evaluates the written report **independently**
- Returns structured feedback:
  - `Score: X/10`
  - Strengths
  - Areas to Improve
  - One-line verdict
- This is the **self-corrective layer** - closes the RAG loop

---

## 🗂️ Project Structure

```
ResearchMind/
│
├── 📄 agents.py          # LangChain agent builders + LLM chains (writer, critic)
├── 🔧 tools.py           # Custom @tool decorators - web_search & scrape_url
├── 🔁 pipeline.py        # Sequential agent pipeline orchestrator
├── 🖥️  ui.py             # Web interface with custom dark-mode design system
├── 🚀 main.py            # Entry point (CLI)
├── 📦 requirement.txt    # All Python dependencies
└── 🔒 .gitignore         # Env vars, caches, generated reports excluded
```

### File Responsibilities at a Glance

| File | Role | Key Tech |
|------|------|----------|
| `agents.py` | Defines LLM agents + prompt chains | LangChain, Groq, ChatPromptTemplate |
| `tools.py` | Web search & scraping tools | Tavily, BeautifulSoup, `@tool` decorator |
| `pipeline.py` | Wires all stages together end-to-end | Sequential execution, state dict |
| `ui.py` | Web interface with pipeline visualizer | Custom CSS, Space Grotesk font |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API Keys for: **Groq**, **Tavily**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SaurabSharma09/Multi-Agent-Research-System
cd ResearchMind

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirement.txt
```

### Environment Setup

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

> 🔑 **Get your keys:**
> - Groq: [console.groq.com](https://console.groq.com) - Free tier available
> - Tavily: [tavily.com](https://tavily.com) - Research-optimized search API

### Run the App

```bash
# Launch the web interface
streamlit run ui.py

# CLI mode
python pipeline.py
```

Then open your browser at `http://localhost:8501` 🎉

---

## 🌐 Live Demo

<div align="center">

### 🔗 [multi-agents-research-system](https://multi-agents-research-system.streamlit.app/)

<br/>

**No setup needed. Just enter a topic and watch 4 agents work in real-time.**

Try these topics:
- `"Quantum computing breakthroughs in 2025"`
- `"CRISPR gene editing latest research"`
- `"Large language models and reasoning"`
- `"Fusion energy progress 2024-2025"`

</div>

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 🤖 **LLM** | Groq + LLaMA 3.3 70B | Ultra-fast, high-quality inference |
| 🔗 **Agent Framework** | LangChain 0.2+ | Agent creation, prompt chaining, tool binding |
| 🌐 **Search** | Tavily Search API | Real-time web search optimized for AI |
| 🕷️ **Scraping** | BeautifulSoup4 + Requests | HTML parsing and content extraction |
| 🖥️ **Frontend** | Python Web UI | Reactive interface with live pipeline updates |
| ⚙️ **Env Mgmt** | python-dotenv | Secure API key management |
| 📊 **Observability** | LangSmith (optional) | Trace agent runs, debug chains |
| 🎨 **UI Design** | Space Grotesk + JetBrains Mono | Custom dark-mode design system |

</div>

---

## 📈 What I Learned

This project was a **week of deep learning** - and here's what actually clicked:

### 🧩 1. Multi-Agent Architecture Is Not Just "More Agents"

Building this taught me that specialization matters more than quantity. Having 4 agents each do *one thing extremely well* outperforms a single mega-agent trying to do everything. The separation of concerns - search, read, write, critique - mirrors how actual research teams work.

### 🔁 2. The Power of Self-Critique in AI Systems

Adding the Critic agent wasn't just a nice-to-have. It changed how I think about LLM output quality. Without external evaluation, you can't know if the generated report is good. The critique loop is the closest you can get to **automated quality assurance** in a RAG pipeline.

### 🌐 3. Retrieval Quality > Generation Quality

I spent hours tweaking the writer prompt, only to realize that the real bottleneck was *what* the agents were retrieving. Switching from generic search to Tavily's research-optimized API improved report quality more than any prompt change. **Garbage in, garbage out.**

### ⚡ 4. Groq Is Genuinely Fast

Moving from OpenAI GPT to Groq + LLaMA 3.3 was a revelation. The 70B model via Groq is faster than GPT-3.5 and produces output comparable to GPT-4 on structured tasks. For a pipeline with 4 LLM calls, this matters enormously.

### 🎨 5. UI/UX Makes Your Project Credible

The web interface isn't just a demo - it's a communication tool. Building the pipeline step visualizer, the live progress bar, and the dark design system forced me to think about *user experience*, not just backend logic. This is what makes a project feel polished vs. just functional.

### 📚 6. Research Papers Are Actually Readable

Reading the CRAG paper (Yan et al., 2024) changed how I approached retrieval. Academic papers feel intimidating at first, but breaking them down section by section reveals the core ideas quickly. The architecture I built is directly inspired by Figure 2 of that paper.

---

## 🔮 Future Roadmap

- [ ] 🔄 **True CRAG Loop** - Add retrieval quality evaluator; trigger re-search if score < threshold
- [ ] 🧠 **LangGraph Migration** - Move pipeline to LangGraph for stateful, conditional branching
- [ ] 💾 **Vector Memory** - Store past reports in a vector DB for cross-topic reasoning
- [ ] 🔢 **Multi-Source Scraping** - Scrape top 3 URLs instead of 1 for richer context
- [ ] 📊 **Report Analytics Dashboard** - Track critic scores, topic trends, agent latencies
- [ ] 🌍 **Multi-language Support** - Research in 10+ languages
- [ ] 🤝 **Human-in-the-Loop** - Let users approve/reject search results before writing

---

## 🙌 Acknowledgements

- **[LangChain](https://github.com/langchain-ai/langchain)** - The backbone of the entire agent system. Their documentation and `create_agent` abstractions made multi-agent orchestration possible.
- **[Groq](https://groq.com/)** - For providing insanely fast LLaMA inference that makes a 4-stage pipeline feel instant.
- **[Tavily](https://tavily.com/)** - For building a search API actually designed for AI agents - not just scraping Google.
- **Yan et al. (2024)** - *Corrective Retrieval Augmented Generation* - the research paper that grounded this project's architecture in real academic work.

---

<div align="center">

**Built with 🔬 research curiosity, ⚡ late nights, and genuine love for AI systems.**

<br/>

⭐ **If this project helped you understand multi-agent RAG systems, a star would mean the world.** ⭐

<br/>

[![Live Demo](https://img.shields.io/badge/Try%20it%20Live-ResearchMind-5EC4A4?style=for-the-badge)](https://multi-agents-research-system.streamlit.app/)

<br/><br/>

*Made with ❤️ | LangChain ·LangChain · Groq · Tavily · Python*

</div>
