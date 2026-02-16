#!/usr/bin/env python3
"""
Test the unified workflow that passes user questions through to the evaluator.
This demonstrates the two-step evaluation process with user's specific questions.
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server import _scrub_with_llm, _evaluate_with_llm

# Check for API key
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("❌ ANTHROPIC_API_KEY not set!")
    print("\nPlease set your API key:")
    print("  export ANTHROPIC_API_KEY='your-key-here'")
    print("\nOr add it to Claude Desktop config at:")
    print("  ~/Library/Application Support/Claude/claude_desktop_config.json")
    exit(1)

# Example candidate text
candidate_text = """
Mike, 35, Father, 5 years of experience engineering, graduated top HBCU,
for the role Lead Engineer at Stripe Data. Led distributed systems projects
at his previous company. He has expertise in Python, Go, and Kubernetes.
Strong problem solver who mentors junior engineers. Has delivered 3 major
projects on time with 99.9% uptime. Holds 2 patents in distributed systems.
"""

# Example user questions
user_questions = [
    "How much should we offer this candidate in terms of salary?",
    "Evaluate this candidate out of 7 on these 5 characteristics: technical skills, leadership, communication, problem-solving, and culture fit",
    "Should we move this candidate to the next interview round? Why or why not?",
]

async def main():
    print("=" * 80)
    print("UNIFIED WORKFLOW TEST: USER QUESTIONS PASSED TO EVALUATOR")
    print("=" * 80)
    print()

    print("ORIGINAL CANDIDATE TEXT:")
    print("-" * 80)
    print(candidate_text.strip())
    print()

    # Step 1: Scrub once
    print("=" * 80)
    print("STEP 1: SCRUBBING PROTECTED CHARACTERISTICS")
    print("=" * 80)
    try:
        scrubbed_text = await _scrub_with_llm(candidate_text)
        print("\nSCRUBBED TEXT:")
        print("-" * 80)
        print(scrubbed_text)
        print()

        # Step 2: Answer each user question
        print("=" * 80)
        print("STEP 2: ANSWERING USER QUESTIONS (with scrubbed text only)")
        print("=" * 80)
        print()

        for i, question in enumerate(user_questions, 1):
            print(f"\n{'=' * 80}")
            print(f"QUESTION {i}: {question}")
            print(f"{'=' * 80}")

            # Evaluate with the user's question
            answer = await _evaluate_with_llm(scrubbed_text, question)

            print("\nANSWER:")
            print("-" * 80)
            print(answer)
            print()

        print("=" * 80)
        print("✅ TEST COMPLETE!")
        print("=" * 80)
        print("\nKey features demonstrated:")
        print("✓ User's specific questions passed to evaluator LLM")
        print("✓ Evaluator answers the exact question asked")
        print("✓ Evaluator receives only scrubbed text (no protected characteristics)")
        print("✓ Single scrubbing followed by multiple evaluations possible")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure:")
        print("1. ANTHROPIC_API_KEY is set correctly")
        print("2. You have API credits available")
        print("3. Network connection is working")

asyncio.run(main())
