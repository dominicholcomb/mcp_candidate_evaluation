# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP server for candidate evaluation that aims to reduce the presence of protected characteristics using a **two-LLM architecture**. The server exposes two tools via FastMCP (stdio transport) for use with Claude Desktop, ChatGPT, or Claude Code.

**Critical Design Decision**: The two-step workflow with visible intermediate data is the ONLY approach. A unified single-tool approach was explicitly removed to ensure transparency and auditability.

## Architecture

Two completely separate LLM calls with no shared conversation history:

1. **LLM 1 (Scrubber)** — `_scrub_with_llm()`: Rewrites protected characteristics (gender, age, race/ethnicity, religion, disability, marital status). Names → "Candidate A/B", "HBCU" → "university", pronouns removed, etc.
2. **LLM 2 (Evaluator)** — `_evaluate_with_llm()`: Receives ONLY scrubbed text + user question. Has ZERO access to original text or scrubbing conversation.

Both tools are defined in `server.py`. Production model: `claude-sonnet-4-5-20250929` for both.

**Three layers control host LLM overclaiming**: (1) tool docstrings, (2) evaluator system prompt with explicit prohibitions, (3) tool output with `SCRUB_DISCLAIMER`, `EVAL_DISCLAIMER`, and `EVAL_HOST_INSTRUCTION` constants.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run MCP server (Claude Desktop launches this automatically)
python3 server.py

# Run test scripts (need API key)
ANTHROPIC_API_KEY='...' python3 evaluation/test_llm_scrubbing.py
ANTHROPIC_API_KEY='...' python3 evaluation/test_unified_workflow.py

# Run bias benchmark (uses .env for API key)
python3 evaluation/benchmark/runner.py --dry-run                    # preview without API calls
python3 evaluation/benchmark/runner.py --n 1 --arms raw_naive       # quick single-arm test
python3 evaluation/benchmark/runner.py --n 30 --concurrency 10      # full run (~$3.50, ~25min)
python3 evaluation/benchmark/runner.py --arms raw_naive raw_matched  # specific arms only

# Execute analysis notebook
cd evaluation/benchmark && jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis_executed.ipynb
```

## Language Rules

**CRITICAL** — applies to all code, docstrings, documentation, and tool output:

- **Never use**: "bias-free", "unbiased", "ensures no bias", "prevents bias", "purely based on", "solely based on", or claims that characteristics "were removed"
- **Instead use**: "aims to reduce", "processed to reduce the presence of", "evaluator receives only scrubbed text", "two separate LLM calls with no shared conversation history"

## Benchmark Framework

Located in `evaluation/benchmark/`. Three-arm design comparing raw LLM API vs. MCP pipeline for demographic bias in hiring selections.

| Arm | Description | What it isolates |
|-----|-------------|-----------------|
| `raw_naive` | Full text with demographics, no system prompt | Baseline LLM bias |
| `raw_matched` | Full text with demographics, evaluator system prompt | Prompt-only effect |
| `mcp` | Text scrubbed via `_scrub_with_llm()`, then evaluated via `_evaluate_with_llm()` | Full pipeline effect |

**Key files:**
- `config.py` — Demographics (4 groups from audit study literature), criteria (3 stereotype tiers), roles, prompt templates, test case generator
- `runner.py` — Async benchmark runner with retry logic, response parsing (including refusal detection), summary statistics, EEOC four-fifths rule check
- `APPROACH.md` — Full methodology document
- `analysis.ipynb` — Jupyter notebook with 14 visualization sections

**Design details:**
- Model: `claude-haiku-4-5-20251001` (benchmark only — cheaper than production Sonnet)
- Runner overrides `server.SCRUBBER_MODEL` and `server.EVALUATOR_MODEL` at import time
- Order counterbalancing: every candidate pair runs in both orderings
- API key loaded from `.env` in project root (gitignored)
- `parse_selection_raw()` distinguishes valid selections, refusals (via `REFUSAL_SIGNALS`), and unparseable responses
- `parse_selection_mcp()` looks for "Candidate A"/"Candidate B" labels

## API Key Handling

No API keys are hardcoded. Two mechanisms:
1. **MCP server**: `ANTHROPIC_API_KEY` env var, set via Claude Desktop config
2. **Benchmark**: `.env` file in project root, auto-loaded by `runner.py`

Both `.env` and `claude_desktop_config.json` are in `.gitignore`.

## Project Structure

```
server.py                           # Main MCP server (FastMCP, 2 tools, 2 LLM functions)
evaluation/
├── test_llm_scrubbing.py           # Basic scrub + evaluate test
├── test_unified_workflow.py        # Multi-question evaluation test
├── evaluation.xml                  # Phase 4 evaluation test cases
└── benchmark/
    ├── config.py                   # Test case definitions and prompt templates
    ├── runner.py                   # Async benchmark runner with CLI
    ├── analysis.ipynb              # Visualization notebook (14 sections)
    ├── APPROACH.md                 # Methodology document
    └── results/                    # JSONL results and PNG figures
```

Test files in `evaluation/` use `sys.path.insert` to import from the parent `server.py`.
