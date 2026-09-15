import requests
import pandas as pd
from intent_definitions import INTENT_DEFINITIONS


REFERENCE_FILE = "results/reference_labeled_100.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"


def load_examples():

    df = pd.read_csv(REFERENCE_FILE)

    examples_by_intent = {}

    for intent in INTENT_DEFINITIONS:

        intent_rows = df[df["intent"] == intent].head(3)

        examples_by_intent[intent] = []

        for _, row in intent_rows.iterrows():

            examples_by_intent[intent].append(
                f'Customer: "{row["customer_message"]}"\n'
                f'Intent: {row["intent"]}'
            )

    return examples_by_intent


EXAMPLES_BY_INTENT = load_examples()


def classify(customer_message):

    intent_text = "\n".join(
        f"- {intent}: {definition}"
        for intent, definition in INTENT_DEFINITIONS.items()
    )

    example_text = ""

    for intent, examples in EXAMPLES_BY_INTENT.items():

        if examples:

            example_text += f"\n### {intent}\n"
            example_text += "\n\n".join(examples)
            example_text += "\n"

    prompt = f"""
You are a customer-support intent classifier for AmazonHelp.

Choose exactly ONE intent from the following list.

{intent_text}

Below are representative historical examples grouped by intent.

{example_text}

Now classify this customer message:

Customer:
{customer_message}

Rules:
1. Choose exactly ONE intent from the intent list.
2. Determine the customer's actual request or problem.
3. Use the historical examples as guidance.
4. Do not choose an intent merely because one word matches.
5. If the message clearly describes a delayed package, use DELIVERY_DELAY.
6. If the message asks where the package is or asks for its current status, use DELIVERY_TRACKING.
7. If the package is marked delivered but the customer did not receive it, use PACKAGE_NOT_RECEIVED.
8. If something went wrong with the delivery itself, use DELIVERY_PROBLEM.
9. If the order has not been shipped/dispatched, use ORDER_DISPATCH.
10. Output ONLY the exact intent name.
11. Do not explain your answer.

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