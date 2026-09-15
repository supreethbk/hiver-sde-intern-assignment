import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

df = pd.read_csv("results/amazonhelp_golden_labeled.csv")

intents = sorted(df["final_intent"].unique())

for intent in intents:
    examples = df[df["final_intent"] == intent]["customer_message"].tolist()

    example_text = "\n".join(
        f"- {x}" for x in examples[:10]
    )

    prompt = f"""
You are helping design an intent taxonomy for Amazon customer support.

Intent name:
{intent}

Examples belonging to this intent:
{example_text}

Write ONE short definition that explains what this intent means
and helps distinguish it from similar intents.

Rules:
- Maximum 30 words.
- Use simple language.
- Focus on the customer's problem or request.
- Do not rename the intent.
- Output only the definition.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )

    definition = response.choices[0].message.content

    print(f"\n{intent}")
    print(definition)