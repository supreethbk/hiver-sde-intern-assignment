import requests
import pandas as pd
from intent_definitions import INTENT_DEFINITIONS


REFERENCE_FILE = "results/reference_labeled_100.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"


def load_examples():
    df = pd.read_csv(REFERENCE_FILE)

    examples = []

    for _, row in df.iterrows():
        examples.append(
            f'Customer: "{row["customer_message"]}"\n'
            f'Intent: {row["intent"]}'
        )

    return "\n\n".join(examples)


REFERENCE_EXAMPLES = load_examples()


def classify(customer_message):

    intent_text = "\n".join(
        f"- {intent}: {definition}"
        for intent, definition in INTENT_DEFINITIONS.items()
    )

    prompt = f"""
You are a customer-support intent classifier for AmazonHelp.

Choose exactly ONE intent from the following list.

{intent_text}

Here are examples from historical AmazonHelp customer conversations.
Use them as guidance for understanding how intents are expressed.

--- HISTORICAL EXAMPLES ---
{REFERENCE_EXAMPLES}
--- END HISTORICAL EXAMPLES ---

Now classify this new customer message.

Customer message:
{customer_message}

Rules:
1. Choose exactly one intent from the intent list.
2. Use the historical examples as guidance, not as an exact text-matching system.
3. Identify the customer's actual request or problem.
4. Output ONLY the exact intent name.
5. Do not explain your answer.

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

    result = result.replace("Intent:", "").strip()

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