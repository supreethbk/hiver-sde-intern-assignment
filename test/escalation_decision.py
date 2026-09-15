# ============================================================
# Escalation Decision Baseline
# ============================================================

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

    # --------------------------------------------------------
    # Always escalate when the intent is too vague.
    # --------------------------------------------------------

    if intent == "INSUFFICIENT_CONTEXT":
        return {
            "decision": "ESCALATE",
            "reason": "Customer message does not contain enough context."
        }

    # --------------------------------------------------------
    # Intents that normally require human/support-system action.
    # --------------------------------------------------------

    if intent in ESCALATE_INTENTS:
        return {
            "decision": "ESCALATE",
            "reason": f"Intent '{intent}' may require account access or support action."
        }

    # --------------------------------------------------------
    # Simple informational / conversational cases.
    # --------------------------------------------------------

    if intent in AUTO_HANDLE_INTENTS:
        return {
            "decision": "AUTO_HANDLE",
            "reason": f"Intent '{intent}' can usually be handled without private account action."
        }

    # --------------------------------------------------------
    # Conditional cases.
    # --------------------------------------------------------

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
                    )
                }

        return {
            "decision": "AUTO_HANDLE",
            "reason": (
                f"Intent '{intent}' appears suitable for a "
                "non-account-action response."
            )
        }

    # --------------------------------------------------------
    # Conservative default.
    # --------------------------------------------------------

    return {
        "decision": "ESCALATE",
        "reason": "Intent is not confidently safe for automatic handling."
    }


# ============================================================
# Smoke tests
# ============================================================

if __name__ == "__main__":

    test_cases = [
        (
            "DELIVERY_TRACKING",
            "You can check your delivery status for the latest update."
        ),
        (
            "REFUND",
            "Please contact Amazon support for assistance with your refund."
        ),
        (
            "CASUAL_ENGAGEMENT",
            "You're welcome! Let us know if you need anything else."
        ),
        (
            "INSUFFICIENT_CONTEXT",
            "Could you provide more details about the issue?"
        ),
        (
            "ORDER_DISPATCH",
            "Your order is still being processed for dispatch."
        ),
    ]

    print("\n========================================")
    print("Escalation Decision Baseline")
    print("========================================")

    for intent, reply in test_cases:

        result = decide_escalation(
            intent,
            reply
        )

        print("\nIntent:", intent)
        print("Reply:", reply)
        print("Decision:", result["decision"])
        print("Reason:", result["reason"])