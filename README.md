# LV Combined Agents

A multi-specialist agent orchestrator built with [DeepAgents](https://docs.langchain.com/oss/python/deepagents/overview).

The orchestrator agent delegates complex tasks to specialized subagents:
- **planner** – Break down complex goals into concrete, ordered steps
- **websearch** – Gather technical context and conduct live web search
- **writer** – Draft polished user-facing text and documentation
- **coder** – Implement code and technical changes safely
- **reviewer** – Critique outputs for bugs, regressions, and missing validation
- **presenter** – Create professional presentation slide decks in PowerPoint (PPTX) and PDF formats
- **ragsub** – Ingest local documents, retrieve + rerank context, and answer with citations

**This example demonstrates:**
- Agent configuration through filesystem primitives (AGENTS.md, skills/, subagents.yaml)
- Multi-agent orchestration and delegation patterns
- Live web search capabilities via Tavily integration
- Cross-session conversation recall via transcript search
- Lightweight self-improvement through reusable learning notes
- Custom tools and environment configuration
- Swappable local/remote providers: Ollama, llama.cpp, Hugging Face
- Presentation generation with python-pptx and reportlab

## Quick Start

### Prerequisites
- Python 3.11+
- `uv` package manager
- Ollama running locally (if using Ollama)
- Local GGUF model file (if using llama.cpp)
- Hugging Face token (if using Hugging Face Inference)
- Tavily API key (for web research) – [Get free tier here](https://www.tavily.com/)

### Setup

1. **Get a Tavily API key** (optional but recommended for research tasks):
   - Visit [tavily.com](https://www.tavily.com/) and sign up
   - Copy your API key

2. **Pull the Ollama model** (first time only):
```powershell
ollama pull devstral-small-2
```
Or use a different model:
```powershell
ollama pull qwen3.5:14b
```

3. **Start Ollama** (keep running):
```powershell
ollama serve
```

4. **Configure environment variables**:
```powershell
# Copy .env.example to .env
cp .env.example .env

# Edit .env and add your Tavily API key (for web research)
# TAVILY_API_KEY=your_key_here

# Or use environment variables directly:
$env:TAVILY_API_KEY="your_key_here"
$env:DEEPAGENT_MODEL="ollama:devstral-small-2"
```

5. **Run the agent**:
```powershell
cd g:\my_deepagents\lv_combined_agents
uv run python agent.py "Build a product roadmap for an AI coding assistant"
```

### Example Tasks
```powershell
uv run python agent.py "Create a 2-week sprint plan for a startup"
uv run python agent.py "Research the latest AI trends and summarize findings"
uv run python agent.py "Write a technical proposal for a new system"
uv run python agent.py "Code review and suggest improvements"
uv run python agent.py "Create a presentation on machine learning with 8 slides"
uv run python agent.py "Index docs in temp_rag_upload and answer questions with citations"
```

## Web Chat Interface

Launch an interactive chat interface with the agent orchestrator using Streamlit:

```powershell
uv run streamlit run streamlit_app.py
```

Then open your browser to **http://localhost:8501**

### Features
- 💬 **Chat History** – Maintain context across multiple turns
- 🧠 **Session Recall** – Search prior saved chats before repeating earlier decisions
- 📝 **Agent Learnings** – Persist compact reusable workflow notes across sessions
- 📊 **Presentations** – Generate and download PowerPoint files directly
- 🔍 **Research** – Get web research results integrated with agent responses
- 📥 **Export** – Download chat history as markdown
- ⚙️ **Settings** – View current model and quick command examples

### Example Prompts
- "Create a 5-slide presentation about AI ethics"
- "Research the latest developments in quantum computing"
- "Design a system architecture for a real-time chat application"
- "Review this code for security issues: [paste code]"
- "Write a 2-week roadmap for an MVP startup product"

### Configuration

## Model Selection
By default, the agent uses `ollama:devstral-small-2` (optimized for tool use). Override with environment variable:

```powershell
# Use a different Ollama model with tool support
$env:DEEPAGENT_MODEL="ollama:qwen3.5:14b"
uv run python agent.py "Your task"

# Use Claude (requires ANTHROPIC_API_KEY)
$env:DEEPAGENT_MODEL="anthropic:claude-3-5-sonnet-latest"
uv run python agent.py "Your task"

# Use GPT-4o (requires OPENAI_API_KEY)
$env:DEEPAGENT_MODEL="openai:gpt-4o"
uv run python agent.py "Your task"

# Use llama.cpp (local GGUF file)
$env:DEEPAGENT_LLM_PROVIDER="llama_cpp"
$env:DEEPAGENT_MODEL="llama_cpp:G:/models/llama-3.1-8b-instruct.Q4_K_M.gguf"
uv run python agent.py "Your task"

# Use Hugging Face Inference endpoint
$env:DEEPAGENT_LLM_PROVIDER="huggingface"
$env:HUGGINGFACEHUB_API_TOKEN="hf_xxx"
$env:DEEPAGENT_MODEL="huggingface:meta-llama/Meta-Llama-3.1-8B-Instruct"
uv run python agent.py "Your task"
```

**Recommended models with tool-calling support for 24GB VRAM:**
- `ollama:devstral-small-2` (24B) – Best for agents with tool use
- `ollama:qwen3.5:14b` (14B) – Conservative VRAM usage
- `anthropic:claude-3-5-sonnet-latest` – Excellent tool use (remote)
- `openai:gpt-4o` – Excellent tool use (remote)

See `.env.example` for required environment variables.

## Project Structure

```
lv_combined_agents/
├── agent.py               # Main orchestrator implementation
├── conversation_memory.py # Transcript search and reusable learning store
├── AGENTS.md              # Shared agent memory and operating context
├── subagents.yaml         # Subagent definitions
├── deepagents.toml        # Agent configuration and deployment settings
├── pyproject.toml         # Python dependencies
├── presentation_tools.py  # Utilities for PowerPoint/PDF generation
├── demo_presentation.py   # Example presentation generation
├── websearch_agent/       # Web search module
│   ├── __init__.py
│   ├── tools.py           # Tavily search and think tools
│   └── prompts.py         # Research instructions
├── skills/
│   ├── execution/
│   │   └── SKILL.md       # Execution workflow skill
│   ├── review/
│   │   └── SKILL.md       # Quality review skill
│   └── presentation/
│       └── SKILL.md       # Presentation creation skill
└── README.md              # This file
```

### File Purposes

| File | Purpose | When Loaded |
|------|---------|------------|
| **AGENTS.md** | Shared memory about mission, delegation rules, and style | On agent initialization |
| **subagents.yaml** | Specialist definitions and their prompts | On agent initialization |
| **skills/** | Reusable workflows for specific tasks | On-demand by the agent |
| **agent.py** | Orchestrator logic and entry point | On execution |
| **conversation_memory.py** | Persists chats, indexes transcripts, and stores reusable learnings | During chat save/search |
| **presentation_tools.py** | Utilities for generating PowerPoint and PDF presentations | When creating presentations |
| **demo_presentation.py** | Example script demonstrating presentation generation | Optional - for testing |

## Presentation Tools

The **Presenter** agent can create professional presentation slide decks in both PowerPoint (PPTX) and PDF formats using the `presentation_tools` module.

### Features
- Generate editable PowerPoint presentations (PPTX)
- Create shareable PDF slide decks
- Consistent styling and professional layouts
- Support for bullet points and text content

### Usage Example

```python
from presentation_tools import create_pptx_presentation, create_pdf_presentation

# Define slides with title and content
slides = [
    {
        "title": "Introduction",
        "content": ["Welcome to the presentation", "Today's agenda"]
    },
    {
        "title": "Key Points",
        "content": ["Point 1", "Point 2", "Point 3"]
    },
    {
        "title": "Conclusion",
        "content": ["Summary", "Q&A"]
    }
]

# Generate both formats
create_pptx_presentation(slides, "output.pptx")
create_pdf_presentation(slides, "output.pdf")
```

### Demo
Run the included demo to see presentations in action:
```powershell
uv run python demo_presentation.py
```

This creates a 10-slide presentation on AI trends in both formats in the `presentations/` folder.

## Web Research

The **Web Search** agent can conduct live web search using Tavily to gather current information.

### Features
- Search the web for information on any topic
- Fetch and process full webpage content
- Strategic research workflow (think → search → assess → iterate)
- Cite sources with inline references [1], [2], [3]
- Works with or without Tavily (gracefully degrades if not configured)

### Setup
1. Get a free Tavily API key: https://www.tavily.com/
2. Add to `.env`:
```env
TAVILY_API_KEY=your_key_here
```

### Usage Examples
```powershell
# Research current topics
uv run python agent.py "What are the latest developments in Foundation Models?"

# Get specific information
uv run python agent.py "Research and compare Claude vs GPT-4 for coding tasks"

# Complex analysis
uv run python agent.py "Analyze current trends in AI safety and alignment research"
```

The websearch agent will automatically use web search to find current, relevant information and provide citations.

## QA And Evaluation

The repo now has a single local QA entrypoint plus a dedicated live RAG evaluation harness.

### Full QA Run

```powershell
python run_qa.py
```

This runs:
- `ruff check .`
- `pytest`
- `python evals/run_rag_eval.py --fail-under 1.0`

If your local RAG runtime is not available yet, skip the live harness temporarily:

```powershell
python run_qa.py --skip-rag-eval
```

### Live RAG Evaluation

```powershell
python evals/run_rag_eval.py
```

The harness:
- ingests a small fixture corpus under isolated projects
- runs representative retrieval queries against the real RAG stack
- scores expected-source hit rate, citation presence, project isolation, and first relevant rank
- captures per-query retrieval diagnostics such as candidate counts, selected files, and token-budget skips
- writes a JSON report under `evals/results/`

You can tune retrieval empirically by varying mode and fetch parameters:

```powershell
python evals/run_rag_eval.py --mode Hybrid --top-k 4 --fetch-k 40
python evals/run_rag_eval.py --mode "Top-K Globally" --top-k 6 --fetch-k 60
python evals/run_rag_eval.py --min-rerank-score 0.1
python evals/run_rag_eval.py --sweep-defaults
```

The built-in sweep compares a small set of retrieval profiles and writes a summary report under `evals/results/` with the best observed configuration for the current local stack.

If the harness reports zero candidates across all cases, inspect the `ingest_results` section of the generated report first. In practice that usually means the local embedding runtime was unavailable during fixture ingestion.

### Existing Agent Evaluation

The orchestrator evaluation flow remains available:

```powershell
python evals/run_eval.py all
```

Use the RAG harness for retrieval-specific tuning, and `run_eval.py` for end-to-end orchestrator behavior.

## Customization

### Add a New Subagent
Edit `subagents.yaml`:
```yaml
my_specialist:
  description: Describe what this specialist does
  system_prompt: |
    You are the my_specialist specialist.
    Your core responsibility...
```

Then reference it in code or AGENTS.md delegation rules.

### Add a New Skill
1. Create a directory: `skills/my-skill/`
2. Add `skills/my-skill/SKILL.md` with front matter:
```markdown
---
name: my-skill
description: What this skill accomplishes
---

# My Skill

## Goal
...

## Workflow
...
```

3. Reference in AGENTS.md or agent.py

### Tune Agent Behavior
- **AGENTS.md** – Adjust mission, delegation rules, and output standards
- **subagents.yaml** – Refine specialist prompts based on results
- **agent.py** – Customize tools, memory, or backend configuration

## Troubleshooting

### "Connection refused" error
Ollama is not running. Start it in a separate terminal:
```powershell
ollama serve
```

### "Module not found" error
Reinstall dependencies:
```powershell
uv sync
```

### Model downloads slowly
Check your internet connection and available disk space. Large models (20B parameters) may take several minutes.

## Learn More

- [DeepAgents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [Ollama Documentation](https://github.com/ollama/ollama)
