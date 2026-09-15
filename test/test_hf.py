import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

INTENTS = [
    "ACCOUNT",
    "CASUAL_ENGAGEMENT",
    "CUSTOMER_SERVICE",
    "DELIVERY_DELAY",
    "DELIVERY_INSTRUCTIONS",
    "DELIVERY_PROBLEM",
    "DELIVERY_TRACKING",
    "DEVICE",
    "FEEDBACK",
    "INSUFFICIENT_CONTEXT",
    "ORDER_CANCELLATION",
    "ORDER_DISPATCH",
    "OTHER",
    "PACKAGE_NOT_RECEIVED",
    "PAYMENT",
    "PRODUCT_INFORMATION",
    "REFUND",
    "RESOLUTION_CONFIRMATION",
    "RETURN",
    "SELLER",
    "WEBSITE_OR_APP"
]

customer_message = "Where is my Amazon package?"

prompt = f"""
You are a customer-support intent classifier.

Classify the customer message into exactly ONE of these intents:

{INTENTS}

Customer message:
{customer_message}

Rules:
- Output ONLY the exact intent name.
- Do not explain your answer.
- Do not output anything else.
"""

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=150
)

print("Customer:", customer_message)
result = response.choices[0].message.content
print("Predicted intent:", result)