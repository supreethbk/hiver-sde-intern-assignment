import os

import requests
from intent_definitions import INTENT_DEFINITIONS


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
SEED = int(os.environ.get("OLLAMA_SEED", "42"))


def classify(customer_message):
    intent_text = "\n".join(
        f"- {intent}: {definition}"
        for intent, definition in INTENT_DEFINITIONS.items()
    )

    prompt = f"""
You are a customer-support intent classifier.

Choose exactly ONE intent from the following list.

{intent_text}

Customer message:
{customer_message}

Rules:
1. Choose only one intent from the list.
2. Output ONLY the exact intent name.
3. Do not explain your answer.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": SEED,
            }
        }
    )

    response.raise_for_status()

    return response.json()["response"].strip()


if __name__ == "__main__":
    message = "Where is my Amazon package?"

    result = classify(message)

    print("Customer:", message)
    print("Predicted intent:", result)
