"""End-to-end AmazonHelp support-agent orchestration."""
from __future__ import annotations

from src.escalation.escalation_decision import decide_escalation
from src.generation.reply_generator import generate_safe_reply
from src.intent.classifier_v2 import classify
from src.retrieval.retriever import HistoricalRetriever


DELIVERY_INTENTS = {
    "DELIVERY_DELAY", "DELIVERY_INSTRUCTIONS", "DELIVERY_PROBLEM",
    "DELIVERY_TRACKING", "ORDER_DISPATCH", "PACKAGE_NOT_RECEIVED",
}


class SupportAgent:
    def __init__(self) -> None:
        self.retriever = HistoricalRetriever()

    def respond(self, customer_message: str) -> dict[str, object]:
        intent = classify(customer_message)
        route = "DELIVERY" if intent in DELIVERY_INTENTS else "NON_DELIVERY"
        cases = self.retriever.retrieve(customer_message, top_k=3)
        reply, violations = generate_safe_reply(customer_message, cases)
        escalation = (
            {
                "decision": "ESCALATE",
                "reason": "The generated draft violated a safety rule; a human should handle it.",
            }
            if violations
            else decide_escalation(intent, reply)
        )
        return {
            "intent": intent,
            "route": route,
            "retrieval_method": self.retriever.method,
            "historical_cases": cases,
            "reply": reply,
            "safety_violations": violations,
            "escalation": escalation,
        }
