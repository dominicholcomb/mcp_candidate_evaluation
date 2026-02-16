#!/usr/bin/env python3
"""
Candidate Evaluation MCP Server (LLM-Based Scrubbing)

This MCP server uses a two-LLM architecture for candidate evaluation:
1. LLM 1 (Scrubber): Intelligently removes and rewrites protected characteristics
2. LLM 2 (Evaluator): Evaluates the scrubbed text with no access to original

The two LLM calls are separate with no shared conversation history.
"""

from typing import Optional, List
from enum import Enum
import os
import json
from pydantic import BaseModel, Field, field_validator, ConfigDict
from mcp.server.fastmcp import FastMCP
import anthropic

# Initialize the MCP server
mcp = FastMCP("candidate_evaluation_mcp")

# Initialize Anthropic async client
# API key should be set in environment variable ANTHROPIC_API_KEY
client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Model configuration
# Using Claude Sonnet 4.5 (released Jan 2025)
SCRUBBER_MODEL = "claude-sonnet-4-5-20250929"  # Fast, accurate for scrubbing
EVALUATOR_MODEL = "claude-sonnet-4-5-20250929"  # Same model for evaluation

# Enums
class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"

class ProtectedCategory(str, Enum):
    """Protected characteristic categories that can be scrubbed."""
    GENDER = "gender"
    AGE = "age"
    RACE_ETHNICITY = "race_ethnicity"
    RELIGION = "religion"
    DISABILITY = "disability"
    MARITAL_STATUS = "marital_status"

# Pydantic Models for Input Validation
class ScrubTextInput(BaseModel):
    """Input model for text scrubbing operations."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    text: str = Field(
        ...,
        description="The candidate text to scrub (e.g., resume, application, profile)",
        min_length=1,
        max_length=50000
    )
    categories: Optional[List[ProtectedCategory]] = Field(
        default=None,
        description="Specific categories to scrub. Options: gender, age, race_ethnicity, religion, disability, marital_status. If not provided, all categories will be scrubbed.",
        max_length=6
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'json' for structured data or 'markdown' for human-readable"
    )

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()

class EvaluateScrubbedInput(BaseModel):
    """Input model for evaluating already-scrubbed candidate text."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    scrubbed_text: str = Field(
        ...,
        description="The ALREADY SCRUBBED candidate text (output from scrub_protected_characteristics tool)",
        min_length=1,
        max_length=50000
    )
    evaluation_criteria: Optional[str] = Field(
        default=None,
        description="Optional: User's evaluation question or specific criteria (e.g., 'How much should we offer?', 'Rate 1-7 on leadership'). If not provided, uses default criteria.",
        max_length=5000
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'json' or 'markdown' (default: 'markdown')"
    )

    @field_validator('scrubbed_text')
    @classmethod
    def validate_scrubbed_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Scrubbed text cannot be empty")
        return v.strip()

# Disclaimers included in tool output so the host LLM sees them at response time
SCRUB_DISCLAIMER = (
    "This text was processed by an LLM to identify and rewrite protected characteristics. "
    "This is a best-effort process that aims to reduce the presence of protected characteristics "
    "but may not catch all instances, particularly implicit or contextual indicators. "
    "Review the scrubbed output before use."
)
EVAL_DISCLAIMER = (
    "This evaluation was performed on text that was processed to reduce the presence of "
    "protected characteristics. This process aims to reduce bias in evaluation but does not "
    "guarantee its elimination. The scrubbing may not catch all protected characteristics, "
    "and the evaluating LLM may itself reflect biases present in its training data. "
    "This evaluation should be treated as one input in a broader decision-making process."
)

# Direct instruction to the host LLM to surface a disclaimer when presenting results
EVAL_HOST_INSTRUCTION = (
    "When presenting this evaluation to the user: (1) Do NOT frame the results using "
    "language like 'based on qualifications only', 'bias-free', 'unbiased', 'purely "
    "based on', or claim that protected characteristics 'were removed'. (2) Include a "
    "single-sentence disclaimer that this process aims to reduce bias but may not "
    "eliminate it, and the evaluating LLM may reflect biases from its training data."
)

# Shared utility functions
def _get_scrubbing_prompt(categories: Optional[List[ProtectedCategory]] = None) -> str:
    """Generate the system prompt for the scrubbing LLM."""

    all_categories = {
        "gender": "Gender references (pronouns, titles, gendered terms)",
        "age": "Age references (numbers, descriptors like 'young' or 'senior')",
        "race_ethnicity": "Race/ethnicity indicators (including implicit ones like 'HBCU', 'ESL', cultural names)",
        "religion": "Religious affiliations and related terms",
        "disability": "Disability status and related terms",
        "marital_status": "Marital status, family status, parental information"
    }

    if categories:
        categories_to_scrub = {cat.value: all_categories[cat.value] for cat in categories}
    else:
        categories_to_scrub = all_categories

    categories_list = "\n".join([f"- **{cat}**: {desc}" for cat, desc in categories_to_scrub.items()])

    return f"""You are a bias-reduction specialist. Your job is to rewrite candidate information to remove protected characteristics while preserving all professionally relevant details.

**Categories to remove:**
{categories_list}

**Important guidelines:**
1. **Rewrite, don't just redact**: Replace biased terms with neutral ones
   - All personal names → "the candidate" (single person) or "Candidate A", "Candidate B", etc. (multiple people). Names can signal gender, race, or ethnicity.
   - "HBCU" → "university"
   - "she led the team" → "led the team"
   - "35-year-old" → remove entirely or replace with "experienced"
   - "father of two" → remove entirely

2. **Preserve all relevant information**: Keep skills, experience, achievements, education credentials
3. **Be thorough**: Catch both explicit and implicit bias indicators
4. **Natural language**: The result should read naturally, not have obvious gaps
5. **Context-aware**: Understand when terms have multiple meanings
6. **Multiple candidates**: If the text contains multiple candidates, replace each person's name with a neutral sequential label (Candidate A, Candidate B, etc.) and preserve the list structure

Return ONLY the scrubbed text, with no explanations or metadata."""

async def _scrub_with_llm(text: str, categories: Optional[List[ProtectedCategory]] = None) -> str:
    """
    Use an LLM to intelligently scrub protected characteristics.

    Args:
        text: The text to scrub
        categories: Optional list of specific categories to scrub

    Returns:
        Scrubbed text with protected characteristics removed/rewritten
    """
    system_prompt = _get_scrubbing_prompt(categories)

    try:
        message = await client.messages.create(
            model=SCRUBBER_MODEL,
            max_tokens=8000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Please scrub the following candidate information:\n\n{text}"
                }
            ]
        )

        return message.content[0].text
    except Exception as e:
        raise RuntimeError(f"Error calling Anthropic API for scrubbing: {str(e)}")

async def _evaluate_with_llm(scrubbed_text: str, user_question: Optional[str] = None) -> str:
    """
    Use a separate LLM instance to evaluate the candidate based on scrubbed text.
    This LLM has NO access to the original text or scrubbing conversation.

    Args:
        scrubbed_text: The already-scrubbed candidate text
        user_question: The user's evaluation question or criteria (optional)

    Returns:
        Evaluation assessment answering the user's question
    """
    default_criteria = """
Evaluate the candidate based on:
1. Relevant skills and technical expertise
2. Demonstrated achievements and measurable results
3. Problem-solving abilities and domain knowledge
4. Experience level and career progression
5. Alignment with role requirements
"""

    # Determine if we have a specific question or use default criteria
    if user_question:
        # User has a specific question - answer it directly
        system_prompt = f"""You are a professional candidate evaluator. You are receiving candidate information that has been processed to reduce the presence of protected characteristics. Evaluate based on the professional qualifications presented.

**User's Question:**
{user_question.strip()}

**Guidelines:**
- Answer the user's question directly and specifically
- Focus only on skills, experience, achievements, and qualifications present in the candidate information
- Do not make assumptions about any missing information
- Be objective and evidence-based
- If the question asks for a rating or score, provide it clearly
- If the question asks for a recommendation (e.g., salary, next steps), provide specific guidance
- Do NOT use absolute language like "unbiased", "bias-free", "purely based on", or "solely based on" when describing this evaluation"""

        user_message = f"Based on this candidate information, please answer the question:\n\n{scrubbed_text}"
    else:
        # No specific question - use default evaluation criteria
        system_prompt = f"""You are a professional candidate evaluator. You are receiving candidate information that has been processed to reduce the presence of protected characteristics. Evaluate based on the professional qualifications presented.

**Evaluation Criteria:**
{default_criteria.strip()}

**Guidelines:**
- Focus only on skills, experience, achievements, and qualifications
- Do not make assumptions about any missing information
- Be objective and evidence-based
- Provide constructive feedback on strengths and areas for development
- Do NOT use absolute language like "unbiased", "bias-free", "purely based on", or "solely based on" when describing this evaluation"""

        user_message = f"Please evaluate this candidate:\n\n{scrubbed_text}"

    try:
        message = await client.messages.create(
            model=EVALUATOR_MODEL,
            max_tokens=4000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        return message.content[0].text
    except Exception as e:
        raise RuntimeError(f"Error calling Anthropic API for evaluation: {str(e)}")

# Tool definitions
@mcp.tool(
    name="scrub_protected_characteristics",
    annotations={
        "title": "Scrub Protected Characteristics (LLM-Based)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True  # Makes API calls
    }
)
async def scrub_protected_characteristics(params: ScrubTextInput) -> str:
    """
    Attempt to identify and rewrite protected characteristics in candidate text using an LLM.

    This tool uses Claude to identify and rewrite protected characteristics
    including gender, age, race/ethnicity, religion, disability, and marital/family status.
    This aims to reduce the presence of protected characteristics in candidate text. It can:
    - Rewrite naturally (e.g., "HBCU" → "university")
    - Catch implicit bias indicators
    - Preserve context and readability

    Args:
        params (ScrubTextInput): Validated input parameters containing:
            - text (str): The candidate text to scrub
            - categories (Optional[List[str]]): Specific categories to scrub (default: all)
            - response_format (str): Output format - 'json' or 'markdown' (default: 'json')

    Returns:
        str: Processed text with a disclaimer noting the limitations of this process.
    """
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return json.dumps({
            "error": "ANTHROPIC_API_KEY environment variable not set. Please configure it in Claude Desktop config."
        })

    try:
        # Scrub the text using LLM
        scrubbed_text = await _scrub_with_llm(params.text, params.categories)

        # Prepare response
        categories_requested = [cat.value for cat in params.categories] if params.categories else ["all"]

        # Format response
        if params.response_format == ResponseFormat.JSON:
            result = {
                "scrubbed_text": scrubbed_text,
                "original_length": len(params.text),
                "scrubbed_length": len(scrubbed_text),
                "categories_requested": categories_requested,
                "disclaimer": SCRUB_DISCLAIMER,
            }
            return json.dumps(result, indent=2)
        else:
            # Markdown format
            lines = [
                "# Text Scrubbing Results (LLM-Based)",
                "",
                "## Statistics",
                f"- **Original length**: {len(params.text)} characters",
                f"- **Scrubbed length**: {len(scrubbed_text)} characters",
                f"- **Categories targeted**: {', '.join(categories_requested)}",
                "",
                "## Scrubbed Text",
                "",
                scrubbed_text,
                "",
                "---",
                "",
                f"**Disclaimer**: {SCRUB_DISCLAIMER}",
            ]
            return "\n".join(lines)

    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool(
    name="evaluate_scrubbed_candidate",
    annotations={
        "title": "Evaluate Scrubbed Candidate",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True  # Makes API calls
    }
)
async def evaluate_scrubbed_candidate(params: EvaluateScrubbedInput) -> str:
    """
    Evaluate a candidate using scrubbed text in a separate LLM context.

    First use `scrub_protected_characteristics` to process the text, then pass the
    scrubbed output to this tool for evaluation.

    Args:
        params (EvaluateScrubbedInput): Validated input parameters containing:
            - scrubbed_text (str): Already-scrubbed candidate text (from scrub tool)
            - evaluation_criteria (Optional[str]): User's evaluation question or criteria
            - response_format (str): Output format - 'json' or 'markdown'

    Returns:
        str: Evaluation based on scrubbed text, with a disclaimer noting limitations.
    """
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return json.dumps({
            "error": "ANTHROPIC_API_KEY environment variable not set. Please configure it in Claude Desktop config."
        })

    try:
        # Evaluate with LLM (NO ACCESS to original text)
        evaluation = await _evaluate_with_llm(params.scrubbed_text, params.evaluation_criteria)

        # Format response
        if params.response_format == ResponseFormat.JSON:
            result = {
                "evaluation": evaluation,
                "scrubbed_text_length": len(params.scrubbed_text),
                "disclaimer": EVAL_DISCLAIMER,
                "important_instruction": EVAL_HOST_INSTRUCTION,
            }
            return json.dumps(result, indent=2)
        else:
            # Markdown format
            lines = [
                "# Candidate Evaluation",
                "",
                "## Evaluation",
                "",
                evaluation,
                "",
                "---",
                "",
                f"*Evaluated {len(params.scrubbed_text)} characters of scrubbed text*",
                "",
                f"**Disclaimer**: {EVAL_DISCLAIMER}",
                "",
                f"**{EVAL_HOST_INSTRUCTION}**",
            ]
            return "\n".join(lines)

    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    mcp.run()
