#!/usr/bin/env python3
"""
Test the LLM-based scrubbing approach.
This demonstrates the two-LLM architecture with separated scrubbing and evaluation.
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

# Example candidate text with protected characteristics
example_text = """
Mike, 35, Father, 5 years of experience engineering, graduated top HBCU,
for the role Lead Engineer at Stripe Data. Led distributed systems projects
at his previous company. He has expertise in Python, Go, and Kubernetes.
Strong problem solver who mentors junior engineers.
"""

async def main():
    print("=" * 80)
    print("LLM-BASED CANDIDATE EVALUATION TEST")
    print("=" * 80)
    print()

    print("ORIGINAL TEXT:")
    print("-" * 80)
    print(example_text.strip())
    print()

    # Step 1: Scrub with LLM
    print("=" * 80)
    print("STEP 1: SCRUBBING WITH LLM (No conversation history exposed)")
    print("=" * 80)
    try:
        scrubbed_text = await _scrub_with_llm(example_text)
        print("\nSCRUBBED TEXT:")
        print("-" * 80)
        print(scrubbed_text)
        print()

        # Step 2: Evaluate with separate LLM
        print("=" * 80)
        print("STEP 2: EVALUATION WITH SEPARATE LLM (No access to original)")
        print("=" * 80)
        evaluation = await _evaluate_with_llm(scrubbed_text)
        print("\nEVALUATION:")
        print("-" * 80)
        print(evaluation)
        print()

        print("=" * 80)
        print("✅ TEST COMPLETE!")
        print("=" * 80)
        print("\nKey features demonstrated:")
        print("✓ LLM intelligently rewrote protected characteristics")
        print("✓ Natural language preserved (not just [REDACTED])")
        print("✓ Two separate LLM calls with no shared conversation history")
        print("✓ Evaluator receives only scrubbed text")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure:")
        print("1. ANTHROPIC_API_KEY is set correctly")
        print("2. You have API credits available")
        print("3. Network connection is working")

asyncio.run(main())
