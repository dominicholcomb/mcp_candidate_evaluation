# Candidate Evaluation MCP Server

A Model Context Protocol (MCP) server that enables **transparent candidate evaluation** with protected characteristics removal, using a two-LLM architecture with visible intermediate steps.

> **Disclaimer**: This tool aims to reduce the presence of protected characteristics in candidate evaluation. It is a best-effort process that may not catch all indicators, and the evaluating LLM may reflect biases from its training data. This should be treated as one input in a broader decision-making process.

## Overview

This MCP server uses **LLM-based scrubbing** to remove protected characteristics then evaluate job candidates in new context windows. The **two-step workflow** aims to make the process transparent and verifiable:

### Two-LLM Architecture (Transparent)

```
Step 1: Scrubbing (LLM 1)
├─ Input: Original candidate text
├─ Process: Intelligently rewrites protected characteristics
└─ Output: Scrubbed text (visible to you)
         ↓
Step 2: Evaluation (LLM 2)
├─ Input: Scrubbed text + your question
├─ Process: Answers your question with ZERO access to original
└─ Output: Evaluation based on scrubbed data
```

### Protected Categories

The scrubber removes:
- **Gender**: Pronouns, titles, gender-specific terms
- **Age**: Age references, birth years, age-related descriptors
- **Race/Ethnicity**: Race, ethnicity indicators (including implicit ones like "HBCU", cultural names)
- **Religion**: Religious affiliations and related terms
- **Disability**: Disability status and related terms
- **Marital/Family Status**: Marital status, family status, parental information

## Why This Approach?

**Transparency**: You see exactly what was scrubbed before evaluation
**Verification**: You can confirm the evaluator only receives scrubbed data
**Auditability**: Each step is logged and visible in Claude Desktop
**Separation**: Two distinct LLM calls with no shared conversation history

## Features

- **LLM-based intelligent scrubbing**: Rewrites naturally (e.g., "HBCU" → "university"), not just [REDACTED]
- **True two-LLM separation**: Evaluator has no conversation history from scrubbing
- **User questions passed through**: Ask specific questions and get direct answers
- **Flexible scrubbing**: Choose which categories to scrub or scrub all
- **Multiple output formats**: JSON for programmatic use, Markdown for human readability

## Installation

### Requirements

- Python 3.10 or higher
- Anthropic API key (default) or OpenAI API key (optional alternative provider)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

To use OpenAI as the LLM provider:
```bash
pip install -e ".[openai]"
```

2. Set your API key:
```bash
export ANTHROPIC_API_KEY='your-key-here'
```

Or configure it in Claude Desktop (see Integration section).

## Usage

### Integration with Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "candidate-evaluation": {
      "command": "python3",
      "args": [
        "/path/to/server.py"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

To use OpenAI instead:
```json
{
  "mcpServers": {
    "candidate-evaluation": {
      "command": "python3",
      "args": [
        "/path/to/server.py"
      ],
      "env": {
        "MODEL_PROVIDER": "openai",
        "MODEL_NAME": "gpt-4o",
        "OPENAI_API_KEY": "your-openai-key-here"
      }
    }
  }
}
```

Restart Claude Desktop, and the tools will be available.

### Two-Step Workflow

The MCP server provides two tools that work together:

#### Step 1: `scrub_protected_characteristics`

**Removes protected characteristics** from candidate text using intelligent LLM-based rewriting.

**Parameters:**
- `text` (required): Candidate text to scrub
- `categories` (optional): Specific categories to scrub (default: all)
- `response_format` (optional): 'json' or 'markdown' (default: 'json')

**Returns:**
```json
{
  "scrubbed_text": "Alex Kim. BS in Computer Science from UC Berkeley...",
  "original_length": 393,
  "scrubbed_length": 297,
  "categories_requested": ["all"]
}
```

#### Step 2: `evaluate_scrubbed_candidate`

**Evaluates the scrubbed candidate** with ZERO access to original text.

**Parameters:**
- `scrubbed_text` (required): ALREADY SCRUBBED candidate text (from Step 1)
- `evaluation_criteria` (optional): Your evaluation question or criteria
- `response_format` (optional): 'json' or 'markdown' (default: 'markdown')

**Returns:**
Detailed evaluation answering your question based only on professional qualifications.

### Example Usage in Claude Desktop

**Your request:**
```
How much should we offer this candidate?

Alex Kim, 29, Korean-American, non-binary (they/them). Diagnosed with autism
spectrum disorder (Level 1). BS in Computer Science from UC Berkeley. 4 years
experience as a backend developer at two startups. Strong in Python, Go, and
Kubernetes. Contributed to 3 open-source projects. Prefers written communication
over meetings. Applying for Senior Software Developer at a remote-first company.
```

**What Claude Desktop does:**

**Step 1 (Visible):**
```
Tool: scrub_protected_characteristics
Request: [original text with protected characteristics]
Response: {
  "scrubbed_text": "The candidate holds a BS in Computer Science from UC
Berkeley. Has 4 years of experience as a backend developer at two startups.
Demonstrates strong proficiency in Python,    Go, and Kubernetes. Has
contributions to 3 open-source projects. Prefers written communication over
meetings. Applying for Senior Software Developer at a remote-first company."
}
```

**Step 2 (Visible):**
```
Tool: evaluate_scrubbed_candidate
Request: {
  "scrubbed_text": "[scrubbed text from Step 1]",
  "evaluation_criteria": "How much should we offer this candidate for
                          a Senior Software Developer role?"
}
Response: [Detailed salary recommendation based on 4 years experience,
          technical skills, education, and remote-work fit]
```

**You see both steps clearly**, confirming the evaluator never saw the protected characteristics!

## Real-World Examples

### Example 1: Salary Recommendation

**Question:** "How much should we offer this candidate?"

**Step 1 - Scrubbing:**
- Removes: "28-year-old female engineer. She is married with one child"
- Keeps: "Graduated from MIT in 2018. 6 years of Python experience"

**Step 2 - Evaluation:**
- Sees only: Education, experience, skills
- Answers: "$140k-160k based on 6 years experience and MIT education"

### Example 2: Multi-Criteria Rating

**Question:** "Evaluate this candidate out of 7 on: technical skills, leadership, communication, problem-solving, and culture fit"

**Step 1 - Scrubbing:**
- Removes: Age, gender, race, disability references
- Keeps: Projects delivered, technologies used, team collaboration examples

**Step 2 - Evaluation:**
```
Technical Skills: 6/7
Leadership: 5/7
Communication: 6/7
Problem-Solving: 7/7
Culture Fit: 6/7

[Detailed reasoning based on scrubbed qualifications...]
```

### Example 3: Yes/No Decision

**Question:** "Should we move this candidate to the next round?"

**Step 1 - Scrubbing:**
- Removes: All protected characteristics
- Keeps: Skills, achievements, experience

**Step 2 - Evaluation:**
```
Yes, recommend moving to next round.

Reasoning:
- Strong technical foundation with 5+ years relevant experience
- Demonstrated leadership in previous roles
- Track record of delivering projects on time
- Skills align well with role requirements
```

## Testing

Run the included test files to verify functionality (requires `ANTHROPIC_API_KEY`):

### Test LLM scrubbing and evaluation:
```bash
python evaluation/test_llm_scrubbing.py
```

### Test unified workflow:
```bash
python evaluation/test_unified_workflow.py
```

## Benchmarking & Statistical Analysis

A controlled benchmark framework measures whether the MCP scrubbing pipeline reduces demographic bias in LLM-based hiring evaluations compared to sending candidate data directly to the LLM. Full methodology is documented in [`evaluation/benchmark/APPROACH.md`](evaluation/benchmark/APPROACH.md).

### Research Question

When candidates have identical qualifications but different protected characteristics (race, gender), does the LLM show demographic preferences in selection and compensation tasks — and does the MCP pipeline reduce these preferences?

### Experimental Design

**Two benchmark versions** exist under `evaluation/benchmark/`:

| Version | Arms | Names | Tasks |
|---------|------|-------|-------|
| **v1** (`runner.py`) | 3-arm (Raw Naive, Raw Matched, MCP) | Fixed names from audit study literature | Selection only |
| **v2** (`runner_v2.py`) | 2-arm (Raw Naive, MCP) | Randomized from demographically-associated pools per trial | Selection + Salary recommendation |

**Demographics**: 2 races (Black/White) x 2 genders (Male/Female) = 4 groups. Names drawn from audit study literature (Bertrand & Mullainathan 2004).

**Three-arm design (v1)** isolates what drives bias reduction:

| Arm | Description | What it isolates |
|-----|-------------|-----------------|
| **Raw Naive** | Full text with demographics, no system prompt | Baseline LLM bias |
| **Raw Matched** | Full text with demographics, evaluator system prompt | Prompt-only effect |
| **MCP Pipeline** | Scrubbed via `_scrub_with_llm()`, then evaluated via `_evaluate_with_llm()` | Full pipeline effect |

**Controls:**
- **Order counterbalancing** — every candidate pair runs in both orderings to control for first-position bias
- **Constant qualifications** — only demographics vary between paired candidates
- **Same model** — all arms use `claude-haiku-4-5-20251001`
- **Default temperature** (1.0) — captures natural LLM variance; N=30 per cell provides statistical power

### Test Case Matrix (v1)

- 6 unique demographic pairs x 2 orderings x 3 criteria tiers x 2 roles = **72 configs per arm**
- 3 arms x 72 configs x N=30 repetitions = **6,480 total trials**

### Evaluation Criteria

Three criteria tiers test interaction with gender stereotypes:

| Tier | Criteria | Purpose |
|------|----------|---------|
| Female-stereotyped | "nurturing and gentle" | Tests gender-stereotype alignment |
| Male-stereotyped | "tough and logical" | Tests gender-stereotype alignment |
| Neutral | "technically proficient" | Baseline without stereotype loading |

### Statistical Methods

- **Binomial test** against 50% for each pairwise comparison
- **Chi-squared test** across all groups within an arm
- **EEOC four-fifths rule** — flags any comparison where selection rate ratio < 0.80 (adverse impact threshold)
- **Wilson score confidence intervals** (95%) for all reported proportions (v2)

### Analysis & Visualization

Results are analyzed in two Jupyter notebooks:

- **[`analysis.ipynb`](evaluation/benchmark/analysis.ipynb)** (v1) — 14 visualization sections including response classification, selection rates by demographic group, race and gender disparity, pairwise comparisons, criteria-stereotype interaction, first-position bias, role interaction, full interaction heatmap, EEOC four-fifths rule check, binomial significance tests, and bias reduction summary
- **[`demo_plots.ipynb`](evaluation/benchmark/demo_plots.ipynb)** (v2) — 3 demo plots with Wilson confidence intervals covering selection rate by race, selection rate by gender, and salary recommendation completion by demographic group

### Running the Benchmark

```bash
# Preview without API calls
python3 evaluation/benchmark/runner.py --dry-run

# Quick single-arm test
python3 evaluation/benchmark/runner.py --n 1 --arms raw_naive

# Full v1 run (~$3.50, ~25 min)
python3 evaluation/benchmark/runner.py --n 30 --concurrency 10

# v2 with randomized names
python3 evaluation/benchmark/runner_v2.py --n 6 --concurrency 10

# Execute analysis notebooks
cd evaluation/benchmark && jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis_executed.ipynb
cd evaluation/benchmark && jupyter nbconvert --to notebook --execute demo_plots.ipynb --output demo_plots_executed.ipynb
```

### Limitations

- Tests one model (Haiku 4.5) at one point in time; production uses Sonnet 4.5
- Names carry multiple signals (race, gender, class) that cannot be fully isolated
- "Names + labels" is the strongest demographic signal; real resumes are more subtle
- Stereotyped criteria represent extreme cases to surface bias; real job criteria are more nuanced
- N=30 per cell detects large effects (~20pp); moderate effects may be missed

## Architecture

### Why Two Separate LLMs?

To reduce the presence of protected characteristics in evaluation, we use completely separate LLM instances with **visible intermediate data**:

```
┌─────────────────────────────────┐
│   Original Candidate Text       │
│  "29, Korean-American,          │
│   non-binary, autistic..."      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│      LLM 1 (Scrubber)           │
│  - Rewrites biased terms        │
│  - No access to user question   │
│  - Returns scrubbed text        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Scrubbed Text (VISIBLE)       │  ← YOU CAN VERIFY THIS
│  "BS from UC Berkeley,          │
│   4 years experience..."        │
└────────────┬────────────────────┘
             │
             ├──────────────────┐
             │                  │
             ▼                  ▼
┌─────────────────────┐  ┌──────────────┐
│  LLM 2 (Evaluator)  │  │ User Question│
│ - ZERO access to    │◄─┤ "How much to │
│   original text     │  │  offer?"     │
│ - ZERO conversation │  └──────────────┘
│   history           │
│ - Answers question  │
│   directly          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│   Evaluation Answer             │
│  "$140k-160k based on           │
│   6 years experience..."        │
└─────────────────────────────────┘
```

### Key Features

1. **Separate LLM calls** - Evaluator has no conversation history from scrubbing
2. **User's question included** - Evaluator receives your specific question with scrubbed data
3. **Specific answers** - Responses address your actual question, not generic evaluations
4. **Visible intermediate data** - You can verify what was scrubbed before evaluation
5. **Auditable process** - Clear separation between scrubbing and evaluation steps

## Provider Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PROVIDER` | `"anthropic"` | LLM provider: `"anthropic"` or `"openai"` |
| `MODEL_NAME` | `"claude-sonnet-4-5-20250929"` | Model for both scrubber and evaluator. **Required** when provider is `"openai"`. |
| `ANTHROPIC_API_KEY` | (none) | Required when `MODEL_PROVIDER` is `"anthropic"` (default) |
| `OPENAI_API_KEY` | (none) | Required when `MODEL_PROVIDER` is `"openai"` |

## Platform Compatibility

This server uses MCP's **stdio transport** (launched as a local subprocess), so it currently works with:
- **Claude Desktop** (macOS, Windows, Linux)
- **Claude Code** (VSCode extension)

Integration with ChatGPT, agentic coding tools, and other MCP clients would require adding HTTP/SSE transport. This may be added in the future if there is interest.

## How It Works

### Traditional Approach:
```
Candidate text → Single LLM → Evaluation
• LLM sees all protected characteristics
• No separation between data and evaluation
• No visibility into what influenced the evaluation
```

### This MCP Server:
```
Candidate text → LLM 1 (Scrub) → [Visible scrubbed text] → LLM 2 (Evaluate) → Answer
✅ Evaluator does not receive protected characteristics
✅ You can verify the scrubbed data before evaluation
✅ Two separate LLM calls with no shared conversation history
✅ Auditable process with visible intermediate steps
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

## Support

If you encounter issues:
1. Check that the MCP server is connected in Claude Desktop
2. Verify your `ANTHROPIC_API_KEY` is set correctly
3. Look for error messages in the tool responses
4. Check the Claude Desktop logs for connection issues

For bugs or feature requests, please open an issue on GitHub.

---

> **Disclaimer**: This tool aims to reduce the presence of protected characteristics in candidate evaluation but does not guarantee their complete removal. The scrubbing process may not catch all implicit or contextual indicators, and the evaluating LLM may itself reflect biases present in its training data. This tool should be treated as one input in a broader decision-making process, not as a substitute for thoughtful human review.
