AUTO_HANDLE_INTENTS = {
    "CASUAL_ENGAGEMENT",
    "FEEDBACK",
    "PRODUCT_INFORMATION",
    "DELIVERY_INSTRUCTIONS",
    "DELIVERY_TRACKING",
}

ESCALATE_INTENTS = {
    "ACCOUNT",
    "PAYMENT",
    "REFUND",
    "ORDER_CANCELLATION",
    "PACKAGE_NOT_RECEIVED",
    "DELIVERY_PROBLEM",
    "DEVICE",
    "SELLER",
    "WEBSITE_OR_APP",
    "CUSTOMER_SERVICE",
    "OTHER",
    "INSUFFICIENT_CONTEXT",
    "ORDER_DISPATCH",
}

CONDITIONAL_INTENTS = {
    "DELIVERY_DELAY",
    "RETURN",
    "RESOLUTION_CONFIRMATION",
}


def decide_escalation(intent, reply):

    if intent == "INSUFFICIENT_CONTEXT":
        return {
            "decision": "ESCALATE",
            "reason": "Customer message does not contain enough context.",
        }

    if intent in ESCALATE_INTENTS:
        return {
            "decision": "ESCALATE",
            "reason": f"Intent '{intent}' may require account access or support action.",
        }

    if intent in AUTO_HANDLE_INTENTS:
        return {
            "decision": "AUTO_HANDLE",
            "reason": f"Intent '{intent}' can usually be handled without private account action.",
        }

    if intent in CONDITIONAL_INTENTS:

        risky_phrases = [
            "refund",
            "cancel",
            "replace",
            "replacement",
            "investigate",
            "contact me",
            "call me",
            "where is my",
            "not received",
            "marked delivered",
            "missing",
        ]

        reply_lower = reply.lower()

        for phrase in risky_phrases:
            if phrase in reply_lower:
                return {
                    "decision": "ESCALATE",
                    "reason": (
                        f"Generated reply contains action-sensitive "
                        f"content: '{phrase}'."
                    ),
                }

        return {
            "decision": "AUTO_HANDLE",
            "reason": (
                f"Intent '{intent}' appears suitable for a "
                "non-account-action response."
            ),
        }

    return {
        "decision": "ESCALATE",
        "reason": "Intent is not confidently safe for automatic handling.",
    }
