import requests
from intent_definitions import INTENT_DEFINITIONS


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"


def classify(customer_message):
    intent_text = "\n".join(
        f"- {intent}: {definition}"
        for intent, definition in INTENT_DEFINITIONS.items()
    )

    prompt = f"""
You are a customer-support intent classifier for AmazonHelp.

Choose exactly ONE intent from the following list.

{intent_text}

Customer message:
{customer_message}

Important rules:
1. Identify what the customer is actually asking for or complaining about.
2. Do not classify based only on individual words.
3. Distinguish between closely related delivery intents:
   - DELIVERY_TRACKING = asking where the package is or its current status.
   - DELIVERY_DELAY = package is late or past the expected delivery date.
   - PACKAGE_NOT_RECEIVED = marked/expected delivered but customer did not receive it.
   - DELIVERY_PROBLEM = something went wrong with the delivery itself.
   - ORDER_DISPATCH = order has not been shipped/dispatched.
4. Choose OTHER when the message does not clearly fit a specific intent.
5. Choose INSUFFICIENT_CONTEXT when there is not enough information to determine the intent.
6. Output ONLY the exact intent name.
7. Do not explain your answer.

Intent:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    result = response.json()["response"].strip()

    # Clean accidental formatting from the model
    result = result.replace("Intent:", "").strip()

    # Make sure the result is one of our valid intents
    if result not in INTENT_DEFINITIONS:
        return "OTHER"

    return result


if __name__ == "__main__":
    test_messages = [
        "Where is my Amazon package?",
        "My package was supposed to arrive yesterday.",
        "It says delivered but I never received it.",
        "My order still hasn't been dispatched.",
    ]

    for message in test_messages:
        print("\nCustomer:", message)
        print("Predicted:", classify(message))