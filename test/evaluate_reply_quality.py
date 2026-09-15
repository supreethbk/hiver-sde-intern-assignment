import pandas as pd
import requests
import json
import re


INPUT_FILE = "results/reply_generation_eval_20.csv"
OUTPUT_FILE = "results/reply_quality_judge_20.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"


# ============================================================
# Judge one reply
# ============================================================

def judge_reply(customer_message, generated_reply):

    prompt = f"""
Evaluate this customer-support reply.

CUSTOMER:
{customer_message}

REPLY:
{generated_reply}

Give a score from 0 to 2 for each item.

RELEVANCE:
0 = unrelated
1 = partly relevant
2 = directly relevant

USEFULNESS:
0 = no useful help
1 = somewhat useful
2 = useful next step

GROUNDING:
0 = not consistent with support pattern
1 = partly consistent
2 = consistent

FACTUAL_SAFETY:
0 = invented unsupported facts or actions
1 = questionable claim
2 = no invented facts

LEAKAGE_SAFETY:
0 = contains historical/private information
1 = minor leakage
2 = no leakage

STYLE:
0 = poor or very verbose
1 = acceptable
2 = concise and professional

Return ONLY this format:

RELEVANCE: number
USEFULNESS: number
GROUNDING: number
FACTUAL_SAFETY: number
LEAKAGE_SAFETY: number
STYLE: number
REASON: one short sentence
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 120
            }
        },
        timeout=120
    )

    response.raise_for_status()

    text = response.json()["response"].strip()

    print("\nRaw judge output:")
    print(text)

    # --------------------------------------------------------
    # Extract scores
    # --------------------------------------------------------

    criteria = [
        "RELEVANCE",
        "USEFULNESS",
        "GROUNDING",
        "FACTUAL_SAFETY",
        "LEAKAGE_SAFETY",
        "STYLE"
    ]

    scores = {}

    for criterion in criteria:

        pattern = rf"{criterion}\s*:\s*([012])"

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            scores[criterion.lower()] = int(
                match.group(1)
            )
        else:
            scores[criterion.lower()] = None

    # --------------------------------------------------------
    # Reason
    # --------------------------------------------------------

    reason_match = re.search(
        r"REASON\s*:\s*(.+)",
        text,
        re.IGNORECASE
    )

    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        reason = "No reason returned."

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    valid = all(
        scores[c.lower()] in [0, 1, 2]
        for c in criteria
    )

    if valid:

        total = sum(
            scores[c.lower()]
            for c in criteria
        )

    else:

        total = None

    return {
        "relevance": scores["relevance"],
        "usefulness": scores["usefulness"],
        "grounding": scores["grounding"],
        "factual_safety": scores["factual_safety"],
        "leakage_safety": scores["leakage_safety"],
        "style": scores["style"],
        "total_score": total,
        "valid_judgment": valid,
        "reason": reason
    }


# ============================================================
# Main
# ============================================================

df = pd.read_csv(INPUT_FILE)

results = []

print("Evaluating reply quality...")
print(f"Examples: {len(df)}")


for _, row in df.iterrows():

    print("\n========================================")
    print(f"Example {row['example_id']}")
    print("========================================")

    customer_message = row["customer_message"]
    generated_reply = row["generated_reply"]

    print("Customer:")
    print(customer_message)

    print("\nReply:")
    print(generated_reply)

    try:

        result = judge_reply(
            customer_message,
            generated_reply
        )

        print("\nScores:")
        print(result)

        results.append({
            "example_id": row["example_id"],
            "customer_message": customer_message,
            "generated_reply": generated_reply,
            **result
        })

    except Exception as e:

        print("ERROR:", e)

        results.append({
            "example_id": row["example_id"],
            "customer_message": customer_message,
            "generated_reply": generated_reply,
            "valid_judgment": False,
            "reason": str(e)
        })


# ============================================================
# Save
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n========================================")
print("Reply Quality Summary")
print("========================================")

valid_df = results_df[
    results_df["valid_judgment"] == True
].copy()

print(
    f"Valid judgments: "
    f"{len(valid_df)}/{len(results_df)}"
)

if len(valid_df) > 0:

    columns = [
        "relevance",
        "usefulness",
        "grounding",
        "factual_safety",
        "leakage_safety",
        "style"
    ]

    for column in columns:

        print(
            f"{column}: "
            f"{valid_df[column].mean():.2f}/2"
        )

    print(
        "\nAverage total score: "
        f"{valid_df['total_score'].mean():.2f}/12"
    )

else:

    print("No valid judgments were produced.")


print(
    f"\nSaved to: {OUTPUT_FILE}"
)