# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Model Context Protocol (MCP) server for candidate evaluation that aims to reduce the presence of protected characteristics using a **transparent two-LLM architecture**. The server is designed for use with Claude Desktop, ChatGPT (with Developer Mode), or Claude Code.

**Critical Design Decision**: The two-step workflow with visible intermediate data is the ONLY approach used. A unified single-tool approach was explicitly removed to ensure transparency and auditability.

## Core Architecture

### Two-LLM Separation Pattern

The system uses **two completely separate LLM calls** with no shared conversation history:

1. **LLM 1 (Scrubber)**: Receives original candidate text → intelligently rewrites protected characteristics → returns scrubbed text
2. **LLM 2 (Evaluator)**: Receives ONLY scrubbed text + user question → provides evaluation → has ZERO access to original

**Why this matters**: Each LLM call is independent. The evaluator cannot infer protected characteristics from conversation history because there is no shared history between the two calls.

### Protected Characteristics Categories

- Gender (pronouns, titles, gendered terms)
- Age (numbers, descriptors like "young"/"senior")
- Race/Ethnicity (including implicit indicators like "HBCU", "ESL", cultural names)
- Religion
- Disability
- Marital/Family Status

### Tool Structure

The server exposes **two MCP tools** (not one):

1. **`scrub_protected_characteristics`**: Takes original text → returns JSON with `scrubbed_text` + disclaimer
2. **`evaluate_scrubbed_candidate`**: Takes scrubbed text + optional `evaluation_criteria` → returns evaluation + disclaimer + host instruction

Users (via Claude Desktop) call these tools sequentially, seeing the scrubbed output before evaluation.

## Language and Legal Considerations

**CRITICAL**: This section governs all code, docstrings, documentation, and tool output in this project.

### Prohibited Language

Never use these terms to describe the system's output or behavior:
- "bias-free", "unbiased", "ensures no bias", "prevents bias"
- "purely based on", "solely based on", "based on qualifications only"
- Claims that protected characteristics "were removed" (implies 100% success)

### Correct Language

Describe what the system **does**, not what it **guarantees**:
- "aims to reduce the presence of protected characteristics"
- "evaluator receives only scrubbed text"
- "two separate LLM calls with no shared conversation history"
- "processed to reduce the presence of protected characteristics"

### Three Layers of Host LLM Influence

The host LLM (Claude Desktop) tends to overclaim about the system's capabilities. We control this at three layers:

1. **Tool docstrings** (persistent — loaded every interaction regardless of tool use): Keep lean. Use factual descriptions, avoid absolute claims. These are a token cost on every interaction.
2. **Evaluator system prompt** (LLM 2 only): Includes explicit prohibition against absolute language like "unbiased", "bias-free", "purely based on", "solely based on".
3. **Tool output** (only when tools are called): Includes `SCRUB_DISCLAIMER`, `EVAL_DISCLAIMER`, and `EVAL_HOST_INSTRUCTION` constants. The host instruction explicitly tells the host LLM not to frame results with absolute language and to include a disclaimer.

**Key limitation**: Tool output is context that *influences* the host LLM, not a script it follows verbatim. The host LLM may still paraphrase or elaborate.

## Development Commands

### Running the Server

**For MCP integration (Claude Desktop/ChatGPT):**
```bash
python3 server.py
```
The server runs via stdio transport. Claude Desktop launches it automatically when configured.

**Testing without MCP:**
```bash
# Test scrubbing + evaluation workflow
ANTHROPIC_API_KEY='your-key' python3 test_llm_scrubbing.py

# Test multiple evaluation questions
ANTHROPIC_API_KEY='your-key' python3 test_unified_workflow.py
```

### Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `mcp>=1.0.0` - FastMCP framework
- `anthropic>=0.39.0` - Claude API client (async)
- `pydantic>=2.0.0` - Input validation
- `httpx>=0.27.0` - HTTP client (required by anthropic)

### Claude Desktop Configuration

The server requires the `ANTHROPIC_API_KEY` environment variable, set via Claude Desktop config:

```json
{
  "mcpServers": {
    "candidate-evaluation": {
      "command": "python3",
      "args": ["/absolute/path/to/server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-api03-..."
      }
    }
  }
}
```

**Location**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

## Code Architecture Details

### API Key Handling

**IMPORTANT**: No API keys are hardcoded. The flow is:
1. User sets key in Claude Desktop config
2. Claude Desktop sets `ANTHROPIC_API_KEY` environment variable when launching server
3. Server reads via `os.environ.get("ANTHROPIC_API_KEY")`

**Security**: The config file with the API key should NEVER be committed to git. The `.gitignore` excludes `claude_desktop_config.json`, `.env`, and `*.key`.

### Pydantic Models

Input validation uses Pydantic v2 with `ConfigDict`:

- `ScrubTextInput`: Validates scrubbing requests (text, categories, response_format)
- `EvaluateScrubbedInput`: Validates evaluation requests (scrubbed_text, evaluation_criteria, response_format)

Both models:
- Strip whitespace via `str_strip_whitespace=True`
- Validate on assignment via `validate_assignment=True`
- Use `@field_validator` decorators for custom validation

### LLM Interaction Functions

Two async utility functions handle Anthropic API calls:

```python
async _scrub_with_llm(text: str, categories: Optional[List[ProtectedCategory]]) -> str
async _evaluate_with_llm(scrubbed_text: str, user_question: Optional[str]) -> str
```

**Key implementation details**:
- Uses `anthropic.AsyncAnthropic` client for async compatibility with FastMCP
- Both use `claude-sonnet-4-5-20250929` model
- Separate `await client.messages.create()` calls (no shared state)
- `_evaluate_with_llm()` adapts system prompt based on whether user_question is provided
- Both evaluator prompt variants include explicit prohibition against absolute language
- Errors are caught and returned as JSON with `{"error": "message"}`

### Disclaimer Constants

Three constants in `server.py` control disclaimer behavior:

- **`SCRUB_DISCLAIMER`**: Included in scrub tool output. Notes this is best-effort and may miss implicit indicators.
- **`EVAL_DISCLAIMER`**: Included in evaluate tool output. Notes limitations of scrubbing and that the evaluating LLM may reflect training data biases.
- **`EVAL_HOST_INSTRUCTION`**: Included in evaluate tool output. Directly instructs the host LLM to (1) not use absolute framing language and (2) include a single-sentence disclaimer about limitations.

### Response Format Handling

Tools support two output formats:
- **JSON** (default for scrubbing): Structured data with metadata and disclaimers
- **Markdown** (default for evaluation): Human-readable formatted output with disclaimers

The `ResponseFormat` enum controls this via the `response_format` parameter.

## Project Structure

```
mcp-server-project-1/
├── server.py                    # Main MCP server (FastMCP)
├── test_llm_scrubbing.py        # Test: basic scrubbing + evaluation
├── test_unified_workflow.py     # Test: multiple evaluation questions
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # Python dependencies
├── README.md                    # User documentation
├── CLAUDE.md                    # This file (Claude Code guidance)
├── .gitignore                   # Git exclusions (secrets, caches, IDE files)
└── evaluation.xml               # Phase 4 evaluation test cases
```

## Model Configuration

Current models (lines 29-30 in server.py):
```python
SCRUBBER_MODEL = "claude-sonnet-4-5-20250929"
EVALUATOR_MODEL = "claude-sonnet-4-5-20250929"
```

Both use Claude Sonnet 4.5. When updating models:
- Check Anthropic API documentation for current model names
- Update both constants (or use same model for both)
- Test with `test_llm_scrubbing.py` to verify compatibility

## Distribution Guidelines

**Safe to distribute**:
- `server.py`, test files, `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`

**NEVER distribute**:
- Claude Desktop config file (contains API key)
- Any file with `ANTHROPIC_API_KEY` hardcoded

**Before distributing**:
1. Verify no secrets: `grep -r "sk-ant-api" .`
2. Optionally create `claude_desktop_config.template.json` with placeholder key

## Tool Visibility in Claude Desktop

When the server is running and connected, Claude Desktop shows:
- Tool 1: "Scrub Protected Characteristics (LLM-Based)"
- Tool 2: "Evaluate Scrubbed Candidate"

Users see both tools execute sequentially with intermediate scrubbed text visible between steps.
