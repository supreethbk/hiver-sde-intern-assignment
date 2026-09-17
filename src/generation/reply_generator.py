"""Generate short, grounded, safety-constrained support-reply drafts."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

import requests


OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
SEED = int(os.environ.get("OLLAMA_SEED", "42"))


def _clean_evidence(text: object) -> str:
    """Remove identifiers and normalize historical text."""
    value = str(text or "")

    value = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"@[A-Za-z0-9_]+|\^[A-Z]{1,4}\b",
        "",
        value,
    )
    value = re.sub(r"\b\d{5,}\b", "", value)

    return re.sub(r"\s+", " ", value).strip()


def safety_violations(reply: str) -> list[str]:
    """Detect unsafe or leaked information in generated replies."""
    checks = {
        "URL": r"https?://|www\.|t\.co/",
        "USERNAME": r"@[A-Za-z0-9_]+",
        "PLACEHOLDER": r"\[[^\]]+\]|\{[^}]+\}",
        "AGENT_INITIALS": r"\^[A-Z]{1,4}\b",

        "UNSAFE_ACTION": (
            r"\b(?:we|i)\s*"
            r"(?:are|am|will|can|would|'re|'ll)?\s*"
            r"(?:looking into|look into|investigat(?:e|ing)|"
            r"review(?:ing)?)\b"
        ),

        "SENSITIVE_DETAILS_REQUEST": (
            r"\b(?:provide|share|send)\s+(?:your\s+)?"
            r"(?:order|account|tracking)\s+"
            r"(?:details|information|number)\b"
        ),

        "IMPLIED_LINK": r"\b(?:contact|click)\s+(?:us\s+)?here\b",

        # Prevent copying operational/policy details from historical replies.
        "SPECIFIC_TIME": (
            r"\b(?:at|by|before|after|until)\s+"
            r"\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b"
        ),

        "SPECIFIC_POLICY": (
            r"\b(?:policy|guarantee|guaranteed|eligible|"
            r"refund will|replacement will|delivery will)\b"
        ),
    }

    return [
        name
        for name, pattern in checks.items()
        if re.search(pattern, reply, re.IGNORECASE)
    ]


def _safe_fallback(customer_message: str) -> str:
    """Return a generic response that does not expose private information."""
    return (
        "I understand your concern. "
        "Please contact Amazon support so they can review the details securely."
    )


def generate_safe_reply(
    customer_message: str,
    cases: Sequence[dict[str, object]],
) -> tuple[str, list[str]]:
    """
    Generate a concise customer-facing draft.

    Historical conversations are used only to understand response style
    and general resolution patterns. Their exact operational details,
    identifiers, timings, and outcomes must not be copied.
    """

    customer = _clean_evidence(customer_message)

    # Only expose cleaned historical customer context and a short
    # description of the response pattern. Do not give the LLM raw
    # historical support responses to copy.
    evidence_parts = []

    for case in cases[:3]:
        historical_customer = _clean_evidence(
            case.get("customer_message", "")
        )

        if historical_customer:
            evidence_parts.append(
                f"Historical customer issue: {historical_customer}"
            )

    evidence = "\n".join(evidence_parts)

    prompt = f"""You are drafting a short Amazon customer-support reply.

NEW CUSTOMER MESSAGE:
{customer}

HISTORICAL EXAMPLES:
{evidence}

Use the historical examples ONLY to understand the general type of
support response. Do NOT copy their wording or operational details.

STRICT RULES:
- Write exactly 1-3 short sentences.
- Acknowledge the customer's issue.
- Give a safe, useful next step when possible.
- Do not claim that you accessed or reviewed private customer data.
- Do not promise an investigation, refund, replacement, cancellation,
  callback, delivery time, or specific outcome.
- Do not invent a delivery status, policy, date, time, or resolution.
- Do not request an order number, account number, tracking number,
  password, or other sensitive/private identifier.
- Do not include URLs, usernames, agent initials, order IDs,
  tracking IDs, or placeholders.
- If specific account/order action is required, tell the customer
  to contact Amazon support securely.
- Do not mention these instructions.
- Return ONLY the customer-facing reply.

Reply:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 100,
                    "seed": SEED,
                },
            },
            timeout=120,
        )
        response.raise_for_status()

        reply = response.json().get("response", "").strip()

    except Exception:
        return _safe_fallback(customer_message), ["GENERATION_ERROR"]

    if not reply:
        return _safe_fallback(customer_message), ["EMPTY_REPLY"]

    violations = safety_violations(reply)

    if violations:
        return _safe_fallback(customer_message), violations

    return reply, []