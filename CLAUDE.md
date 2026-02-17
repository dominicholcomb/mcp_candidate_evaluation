# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP server for candidate evaluation that aims to reduce the presence of protected characteristics using a **two-LLM architecture**. The server exposes two tools via FastMCP (stdio transport) for use with Claude Desktop, ChatGPT, or Claude Code.

**Critical Design Decision**: The two-step workflow with visible intermediate data is the ONLY approach. A unified single-tool approach was explicitly removed to ensure transparency and auditability.

## Architecture

Two completely separate LLM calls with no shared conversation history:

1. **LLM 1 (Scrubber)** — `_scrub_with_llm()`: Rewrites protected characteristics (gender, age, race/ethnicity, religion, disability, marital status). Names → "Candidate A/B", "HBCU" → "university", pronouns removed, etc.
2. **LLM 2 (Evaluator)** — `_evaluate_with_llm()`: Receives ONLY scrubbed text + user question. Has ZERO access to original text or scrubbing conversation.

Both tools are defined in `server.py`. Default production model: `claude-sonnet-4-5-20250929` (Anthropic). Users can configure an alternative model or provider (OpenAI) via environment variables.

**Three layers control host LLM overclaiming**: (1) tool docstrings, (2) evaluator system prompt with explicit prohibitions, (3) tool output with `SCRUB_DISCLAIMER`, `EVAL_DISCLAIMER`, and `EVAL_HOST_INSTRUCTION` constants.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install with OpenAI support (optional)
pip install -e ".[openai]"

# Run MCP server (Claude Desktop launches this automatically)
python3 server.py

# Run test scripts (need API key)
ANTHROPIC_API_KEY='...' python3 evaluation/test_llm_scrubbing.py
ANTHROPIC_API_KEY='...' python3 evaluation/test_unified_workflow.py

# Run bias benchmark v1 (uses .env for API key)
python3 evaluation/benchmark/runner.py --dry-run                    # preview without API calls
python3 evaluation/benchmark/runner.py --n 1 --arms raw_naive       # quick single-arm test
python3 evaluation/benchmark/runner.py --n 30 --concurrency 10      # full run (~$3.50, ~25min)
python3 evaluation/benchmark/runner.py --arms raw_naive raw_matched  # specific arms only

# Run bias benchmark v2 (randomized names, adds salary task, raw_naive + mcp only)
python3 evaluation/benchmark/runner_v2.py --dry-run
python3 evaluation/benchmark/runner_v2.py --n 6 --concurrency 10

# Execute analysis notebooks
cd evaluation/benchmark && jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis_executed.ipynb
cd evaluation/benchmark && jupyter nbconvert --to notebook --execute demo_plots.ipynb --output demo_plots_executed.ipynb
```

## Language Rules

**CRITICAL** — applies to all code, docstrings, documentation, and tool output:

- **Never use**: "bias-free", "unbiased", "ensures no bias", "prevents bias", "purely based on", "solely based on", or claims that characteristics "were removed"
- **Instead use**: "aims to reduce", "processed to reduce the presence of", "evaluator receives only scrubbed text", "two separate LLM calls with no shared conversation history"

## Benchmark Framework

Located in `evaluation/benchmark/`. Compares raw LLM API vs. MCP pipeline for demographic bias in hiring selections.

**Two runners exist:**

- `runner.py` (v1) — Three-arm design with fixed candidate names:

  | Arm | Description | What it isolates |
  |-----|-------------|-----------------|
  | `raw_naive` | Full text with demographics, no system prompt | Baseline LLM bias |
  | `raw_matched` | Full text with demographics, evaluator system prompt | Prompt-only effect |
  | `mcp` | Text scrubbed via `_scrub_with_llm()`, then evaluated via `_evaluate_with_llm()` | Full pipeline effect |

- `runner_v2.py` — Two-arm design (`raw_naive` + `mcp` only) with randomized names per trial from demographically-associated pools and an additional salary recommendation task.

**Key files:**
- `config.py` — v1 demographics (4 groups, fixed names from audit study literature), criteria (3 stereotype tiers), roles, prompt templates, test case generator
- `config_v2.py` — v2 demographics with randomized name pools per trial, salary task templates
- `analysis.ipynb` — Jupyter notebook with 14 visualization sections (v1 results)
- `demo_plots.ipynb` — Jupyter notebook with 3 demo plots + Wilson CIs (v2 results)
- `APPROACH.md` — Full methodology document

**Design details:**
- Model: `claude-haiku-4-5-20251001` (benchmark only — cheaper than production Sonnet)
- Both runners override `server.SCRUBBER_MODEL` and `server.EVALUATOR_MODEL` at import time
- Order counterbalancing: every candidate pair runs in both orderings
- API key loaded from `.env` in project root (gitignored)
- `parse_selection_raw()` distinguishes valid selections, refusals (via `REFUSAL_SIGNALS`), and unparseable responses
- `parse_selection_mcp()` looks for "Candidate A"/"Candidate B" labels

## Provider Configuration

Three optional environment variables control the LLM backend (set in Claude Desktop config's `env` block):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PROVIDER` | `"anthropic"` | LLM provider: `"anthropic"` or `"openai"` |
| `MODEL_NAME` | `"claude-sonnet-4-5-20250929"` | Model for both scrubber and evaluator. **Required** (no default) when provider is `"openai"`. |
| `OPENAI_API_KEY` | (none) | Required when `MODEL_PROVIDER` is `"openai"` |

OpenAI is an optional dependency (`pip install openai>=1.0.0` or `pip install -e ".[openai]"`). The `openai` package is imported lazily only when `MODEL_PROVIDER` is `"openai"`.

LLM calls are routed through `_call_llm()`, which abstracts the Anthropic/OpenAI API differences. The client is lazy-initialized via `_get_llm_client()` on first use. Provider config is validated at tool call time via `_validate_provider_config()` so errors surface as MCP tool responses.

## API Key Handling

No API keys are hardcoded. Two mechanisms:
1. **MCP server**: API key env var set via Claude Desktop config (`ANTHROPIC_API_KEY` for Anthropic, `OPENAI_API_KEY` for OpenAI)
2. **Benchmark**: `.env` file in project root, auto-loaded by runners (always uses Anthropic)

Both `.env` and `claude_desktop_config.json` are in `.gitignore`.

## Evaluation Test Data

`evaluation/evaluation.xml` contains QA pairs for testing the MCP server's scrubbing behavior (expected redaction counts, category lists, boolean checks on output). These are deterministic tests against the LLM-based scrubber.

## Import Pattern

`server.py` is the single-file MCP server (entry point: `mcp.run()`). Test files and benchmark runners in `evaluation/` use `sys.path.insert` to import functions and module-level constants directly from `server.py`. The benchmark runners mutate `server.SCRUBBER_MODEL` and `server.EVALUATOR_MODEL` after import to swap in cheaper models. Benchmark runners always use the Anthropic provider regardless of `MODEL_PROVIDER` setting, since they create their own `anthropic.AsyncAnthropic()` client for raw arms.
