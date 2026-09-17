"""Ollama-backed AmazonHelp intent classifier."""
from __future__ import annotations

import os

import requests

from src.intent.intent_definitions import INTENT_DEFINITIONS


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
SEED = int(os.environ.get("OLLAMA_SEED", "42"))


def classify(customer_message: str) -> str:
    """Return one taxonomy intent, using OTHER for invalid model output."""
    intent_text = "\n".join(
        f"- {intent}: {definition}"
        for intent, definition in INTENT_DEFINITIONS.items()
    )
    prompt = f"""You are a customer-support intent classifier for AmazonHelp.

Choose exactly ONE intent from the following list.

{intent_text}

Customer message:
{customer_message}

Rules:
1. Identify the customer's actual request or problem, not isolated words.
2. DELIVERY_TRACKING means a current location or status request.
3. DELIVERY_DELAY means late or past the promised delivery date.
4. PACKAGE_NOT_RECEIVED means marked delivered but not received.
5. DELIVERY_PROBLEM means an unacceptable attempted or completed delivery.
6. ORDER_DISPATCH means not shipped or dispatched.
7. Output ONLY the exact intent name. Do not explain.
"""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": SEED},
        },
        timeout=120,
    )
    response.raise_for_status()
    result = response.json().get("response", "").strip()
    result = result.removeprefix("Intent:").strip()
    return result if result in INTENT_DEFINITIONS else "OTHER"
