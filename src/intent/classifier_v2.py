"""AmazonHelp intent classifier with hierarchical rules and Qwen fallback."""

from __future__ import annotations

import os
import requests

from src.intent.intent_definitions import INTENT_DEFINITIONS


SEED = int(os.environ.get("OLLAMA_SEED", "42"))

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "qwen2.5:7b",
)


def _ask(prompt: str) -> str:
    """Send a prompt to Ollama."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "seed": SEED,
                },
            },
            timeout=120,
        )

        if response.status_code != 200:
            return ""

        return response.json().get("response", "").strip()

    except requests.RequestException:
        return ""


def _clean_result(result: str) -> str:
    """Convert model output into one valid intent."""

    result = result.strip()

    result = result.replace("Intent:", "").strip()
    result = result.replace("FINAL ANSWER:", "").strip()

    if result in INTENT_DEFINITIONS:
        return result

    upper = result.upper()

    for intent in INTENT_DEFINITIONS:
        if upper.startswith(intent):
            return intent

    return "OTHER"


def _recover_other(customer_message: str) -> str:
    """Second-pass classification for OTHER."""

    specific_intents = "\n".join(
        f"- {intent}: {definition}"
        for intent, definition in INTENT_DEFINITIONS.items()
        if intent != "OTHER"
    )

    prompt = f"""
You are reviewing an Amazon customer-support message.

The first classifier returned OTHER.

CUSTOMER MESSAGE:
{customer_message}

INTENTS:
{specific_intents}

Choose a specific intent whenever there is enough evidence.

Important boundaries:

CUSTOMER_SERVICE:
The customer is asking Amazon for help, action, explanation,
contact, follow-up, or escalation.

RESOLUTION_CONFIRMATION:
The customer explicitly confirms that a previous problem was solved.

CASUAL_ENGAGEMENT:
Simple greeting, thanks, appreciation, or casual conversation
without a support issue.

INSUFFICIENT_CONTEXT:
The message is too short or vague to determine the actual issue.

DELIVERY_TRACKING:
The customer wants package location, tracking, expected arrival,
or delivery status.

DELIVERY_DELAY:
The customer says the delivery is already late, delayed,
overdue, or missed its promised date.

PACKAGE_NOT_RECEIVED:
The package was marked/expected delivered but was not received,
or the customer explicitly says the package/order is missing.

DELIVERY_PROBLEM:
Failed delivery, wrong delivery location, unsafe placement,
or incorrect delivery handling.

DELIVERY_INSTRUCTIONS:
Setting, changing, or clarifying delivery instructions.

ORDER_DISPATCH:
The order has not shipped/dispatched.

DEVICE:
Problem or question specifically about an Amazon device.

PRODUCT_INFORMATION:
Question about product features, specifications,
availability, or product details.

FEEDBACK:
Opinion, criticism, praise, review, or feedback.

ACCOUNT:
Account access, login, security, or account settings.

PAYMENT:
Payment, billing, charge, or payment-method issue.

REFUND:
Refund status or refund problem.

RETURN:
Returning an item or return process.

ORDER_CANCELLATION:
Cancelling an existing order.

SELLER:
Problem or interaction specifically involving a seller.

WEBSITE_OR_APP:
Website, application, page, link, or site functionality problem.

OTHER:
Use only when no specific intent fits.

Return ONLY ONE exact intent name.
"""

    return _clean_result(_ask(prompt))


def _llm_classify(customer_message: str) -> str:
    """Main Qwen classification."""

    definitions = "\n".join(
        f"{intent}: {definition}"
        for intent, definition in INTENT_DEFINITIONS.items()
    )

    prompt = f"""
You are an expert Amazon customer-support intent classifier.

Choose EXACTLY ONE intent.

INTENTS:
{definitions}

==================================================
IMPORTANT CLASSIFICATION RULES
==================================================

1. CUSTOMER_SERVICE

Use CUSTOMER_SERVICE when the customer is mainly asking Amazon
for help, action, contact, explanation, follow-up, or escalation.

If the message complains about an unresolved support interaction
but does not clearly identify another specific issue, use
CUSTOMER_SERVICE.

--------------------------------------------------

2. RESOLUTION_CONFIRMATION

Use RESOLUTION_CONFIRMATION only when the customer clearly indicates
that a previous problem has been solved, completed, fixed, or resolved.

Examples:
- "Issue is resolved"
- "Problem solved"
- "Got it resolved"
- "Yes, that worked"

A simple "Thanks" can be RESOLUTION_CONFIRMATION when it clearly
continues a support interaction.

--------------------------------------------------

3. CASUAL_ENGAGEMENT

Use CASUAL_ENGAGEMENT for genuine casual conversation, greetings,
simple social interaction, or appreciation where there is no
support issue.

Do not use CASUAL_ENGAGEMENT when the message clearly confirms
that a support problem was solved.

--------------------------------------------------

4. INSUFFICIENT_CONTEXT

Use INSUFFICIENT_CONTEXT when the message is genuinely too vague
to identify what the customer needs.

Examples:
- "Yesterday."
- "November 7th."
- "Amazon EU."
- An isolated order number.

However, if the surrounding wording itself clearly identifies
a support issue, use that specific intent.

--------------------------------------------------

5. DELIVERY_TRACKING vs DELIVERY_DELAY

DELIVERY_TRACKING:
The customer asks about current location, tracking, status,
expected arrival, or delivery date.

DELIVERY_DELAY:
The customer explicitly says the package is already late,
delayed, overdue, or missed the promised delivery date.

Examples:

"Where is my package?"
→ DELIVERY_TRACKING

"When will my package arrive?"
→ DELIVERY_TRACKING

"It was supposed to arrive yesterday."
→ DELIVERY_DELAY

"My package is 3 days late."
→ DELIVERY_DELAY

If the message explicitly says the delivery is late/delayed,
prefer DELIVERY_DELAY even if tracking information is also mentioned.

--------------------------------------------------

6. PACKAGE_NOT_RECEIVED

Use PACKAGE_NOT_RECEIVED when the customer says that an expected
or marked-delivered package was not received.

Examples:
- "It says delivered but I don't have it."
- "I never received my package."
- "My package is missing."

Do not use PACKAGE_NOT_RECEIVED merely because the customer asks
where a package is.

--------------------------------------------------

7. DELIVERY_PROBLEM

Use DELIVERY_PROBLEM for failed or incorrect delivery handling.

Examples:
- delivery attempt failed
- courier refused delivery
- package delivered incorrectly
- unsafe delivery
- wrong delivery handling

--------------------------------------------------

8. DELIVERY_INSTRUCTIONS

Use DELIVERY_INSTRUCTIONS when the customer is setting,
changing, or discussing delivery instructions/preferences.

Examples:
- safe place
- leave package somewhere
- delivery preference
- evening delivery instruction

If the message is instead complaining that a courier failed
to follow the delivery arrangement, consider DELIVERY_PROBLEM.

--------------------------------------------------

9. ORDER_DISPATCH

Use ORDER_DISPATCH when the issue is specifically that an order
has not shipped, dispatched, or been processed for shipment.

Do not use it merely because the customer discusses delivery.

--------------------------------------------------

10. DEVICE vs PRODUCT_INFORMATION

DEVICE:
The issue concerns using, operating, or troubleshooting an Amazon
device such as Echo, Kindle, Fire device, etc.

PRODUCT_INFORMATION:
The customer wants information about a product, such as features,
availability, specifications, price, or options.

If the message asks how an Amazon device works, prefer DEVICE.

--------------------------------------------------

11. FEEDBACK vs CASUAL_ENGAGEMENT

FEEDBACK:
The customer expresses an opinion, criticism, praise, review,
or specific feedback about Amazon/service/product.

CASUAL_ENGAGEMENT:
Simple social interaction without substantive feedback.

--------------------------------------------------

12. RETURN vs REFUND

RETURN:
The customer wants to return an item or asks about the return process.

REFUND:
The customer is specifically asking about receiving money back,
refund status, or a refund problem.

--------------------------------------------------

13. ACCOUNT vs WEBSITE_OR_APP

ACCOUNT:
Login, account access, security, or account settings.

WEBSITE_OR_APP:
A website/app/page/link is broken, unavailable, or malfunctioning.

If the website works but requires login to access an account,
consider ACCOUNT.

--------------------------------------------------

14. OTHER

Use OTHER only when none of the specific intents reasonably fits.

Do not use OTHER simply because the message is short.

==================================================

CUSTOMER MESSAGE:
{customer_message}

Return ONLY the exact intent name.
"""

    return _clean_result(_ask(prompt))


def classify(customer_message: str) -> str:
    """Classify one AmazonHelp customer message."""

    text = customer_message.lower().strip()

    # =========================================================
    # HIGH-CONFIDENCE RESOLUTION / UNRESOLVED
    # =========================================================

    unresolved_phrases = [
        "still unresolved",
        "not resolved",
        "unresolved",
        "still not resolved",
        "hasn't been resolved",
        "has not been resolved",
        "still waiting",
        "nothing has been resolved",
    ]

    if any(p in text for p in unresolved_phrases):
        return "CUSTOMER_SERVICE"

    resolution_phrases = [
        "issue is resolved",
        "problem is resolved",
        "issue has been resolved",
        "problem has been resolved",
        "resolved now",
        "it is resolved",
        "it's resolved",
        "all resolved",
        "got it resolved",
        "this is resolved",
        "problem solved",
        "issue solved",
        "solved now",
        "got it fixed",
        "it worked",
    ]

    if any(p in text for p in resolution_phrases):
        return "RESOLUTION_CONFIRMATION"

    # =========================================================
    # HIGH-CONFIDENCE PACKAGE NOT RECEIVED
    # =========================================================

    delivered_phrases = [
        "marked delivered",
        "shows delivered",
        "status as delivered",
        "says delivered",
        "marked as delivered",
        "delivered to resident",
    ]

    missing_phrases = [
        "didn't receive",
        "did not receive",
        "haven't received",
        "have not received",
        "not received",
        "never received",
        "wasn't received",
        "was not received",
        "package is missing",
        "package missing",
        "order is missing",
        "order missing",
        "package is lost",
        "package lost",
        "order is lost",
        "order lost",
    ]

    if (
        any(p in text for p in delivered_phrases)
        and any(p in text for p in missing_phrases)
    ):
        return "PACKAGE_NOT_RECEIVED"

    if any(p in text for p in missing_phrases):
        return "PACKAGE_NOT_RECEIVED"

    # =========================================================
    # DELIVERY INSTRUCTIONS
    # =========================================================

    delivery_instruction_phrases = [
        "delivery instructions",
        "delivery instruction",
        "delivery preference",
        "delivery preferences",
        "safe place",
        "safeplace",
        "leave it at",
        "leave package at",
        "leave the package at",
        "not home before",
        "not home until",
        "when i'm not home",
        "when i am not home",
        "deliver after",
        "deliver before",
    ]

    if any(p in text for p in delivery_instruction_phrases):
        return "DELIVERY_INSTRUCTIONS"

    # =========================================================
    # DELIVERY PROBLEM
    # =========================================================

    delivery_problem_phrases = [
        "tried to deliver",
        "delivery attempt",
        "delivery attempts",
        "failed to deliver",
        "couldn't deliver",
        "could not deliver",
        "can't deliver",
        "cannot deliver",
        "redeliver",
        "redelivery",
        "courier refused",
        "driver refused",
        "delivered to the wrong",
        "wrong address",
        "wrong location",
        "left it in the wrong",
        "unsafe delivery",
    ]

    if any(p in text for p in delivery_problem_phrases):
        return "DELIVERY_PROBLEM"

    # =========================================================
    # DELIVERY DELAY
    # =========================================================

    explicit_delay_phrases = [
        "package is late",
        "package is delayed",
        "delivery is late",
        "delivery is delayed",
        "already late",
        "already delayed",
        "days delayed",
        "days late",
        "3 days delayed",
        "past the delivery date",
        "past the expected date",
        "past expected delivery date",
        "past the expected delivery date",
        "overdue",
        "supposed to arrive yesterday",
        "supposed to arrive today",
        "was supposed to arrive yesterday",
        "was supposed to arrive today",
        "delivery was supposed to be yesterday",
        "delivery was supposed to be today",
        "delivery date was supposed to be yesterday",
        "delivery date was supposed to be today",
        "still hasn't arrived",
        "still has not arrived",
        "hasn't arrived yet",
        "has not arrived yet",
    ]

    if any(p in text for p in explicit_delay_phrases):
        return "DELIVERY_DELAY"

    # =========================================================
    # DELIVERY TRACKING
    # =========================================================

    tracking_phrases = [
        "where is my package",
        "where's my package",
        "track my package",
        "tracking my package",
        "where is my order",
        "where's my order",
        "order tracking",
        "package tracking",
        "track the package",
        "track the order",
        "is it coming today",
        "coming today",
        "will it arrive today",
        "arriving today",
        "when will it arrive",
        "when is it coming",
        "expected delivery",
        "delivery date",
        "estimated delivery",
        "delivery status",
        "shipping status",
        "current status",
        "current location",
        "where is it",
    ]

    if any(p in text for p in tracking_phrases):
        return "DELIVERY_TRACKING"

    # =========================================================
    # ORDER CANCELLATION
    # =========================================================

    cancellation_phrases = [
        "cancel my order",
        "cancel the order",
        "want to cancel",
        "need to cancel",
        "how do i cancel",
        "can i cancel",
        "please cancel",
    ]

    if any(p in text for p in cancellation_phrases):
        return "ORDER_CANCELLATION"

    # =========================================================
    # REFUND
    # =========================================================

    refund_phrases = [
        "where is my refund",
        "refund status",
        "waiting for my refund",
        "refund hasn't",
        "refund has not",
        "get a refund",
        "need a refund",
        "want a refund",
        "refund back",
        "money back",
    ]

    if any(p in text for p in refund_phrases):
        return "REFUND"

    # =========================================================
    # RETURN
    # =========================================================

    return_phrases = [
        "return this item",
        "return my item",
        "want to return",
        "how do i return",
        "how can i return",
        "return an item",
        "need to return",
        "can i return",
    ]

    if any(p in text for p in return_phrases):
        return "RETURN"

    # =========================================================
    # PAYMENT
    # =========================================================

    payment_phrases = [
        "payment failed",
        "payment issue",
        "payment problem",
        "payment method",
        "charged twice",
        "wrong charge",
        "billing issue",
        "billing problem",
        "charged me",
        "charge on my",
    ]

    if any(p in text for p in payment_phrases):
        return "PAYMENT"

    # =========================================================
    # ACCOUNT
    # =========================================================

    account_phrases = [
        "can't login",
        "cannot login",
        "can't log in",
        "cannot log in",
        "login problem",
        "login issue",
        "account locked",
        "account access",
        "my account",
        "account settings",
        "account security",
        "password",
        "sign in",
        "signin",
    ]

    if any(p in text for p in account_phrases):
        return "ACCOUNT"

    # =========================================================
    # FALLBACK TO QWEN 7B
    # =========================================================

    result = _llm_classify(customer_message)

    # =========================================================
    # OTHER RECOVERY
    # =========================================================

    if result == "OTHER":
        recovered = _recover_other(customer_message)

        if recovered != "OTHER":
            return recovered

    return result