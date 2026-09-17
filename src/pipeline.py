"""Command-line demo for the AmazonHelp support agent."""
from __future__ import annotations

import argparse

from src.agent.support_agent import SupportAgent


def run() -> None:
    parser = argparse.ArgumentParser(description="Run the AmazonHelp support-agent demo.")
    parser.add_argument("message", nargs="*", help="Incoming customer message")
    args = parser.parse_args()
    message = " ".join(args.message).strip() or input("Customer message: ").strip()
    if not message:
        raise SystemExit("A customer message is required.")

    result = SupportAgent().respond(message)
    print("\n=== AmazonHelp support-agent demo ===")
    print(f"Intent: {result['intent']}")
    print(f"Route: {result['route']}")
    print(f"Retrieval: {result['retrieval_method']}")
    print("\nHistorical evidence:")
    for number, case in enumerate(result["historical_cases"], start=1):
        print(f"{number}. similarity={case['similarity']}")
        print(f"   Customer: {case['customer_message']}")
        print(f"   Historical response: {case['support_response']}")
    print("\nDraft reply:")
    print(result["reply"])
    print("\nSafety check:", "PASS" if not result["safety_violations"] else result["safety_violations"])
    print("Escalation:", result["escalation"]["decision"])
    print("Reason:", result["escalation"]["reason"])
