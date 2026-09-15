"""Evaluate escalation decisions on generated-reply examples."""

import sys
from pathlib import Path

import pandas as pd


# Allow importing from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.escalation.escalation_decision import decide_escalation

INPUT_FILE = Path("results/reply_generation_eval_20.csv")
OUTPUT_FILE = Path("results/escalation_results_reply_20.csv")


def main():
    df = pd.read_csv(INPUT_FILE)

    results = []

    for _, row in df.iterrows():

        result = decide_escalation(
            row["golden_intent"],
            row["generated_reply"],
        )

        results.append({
            "example_id": row["example_id"],
            "golden_intent": row["golden_intent"],
            "generated_reply": row["generated_reply"],
            "decision": result["decision"],
            "reason": result["reason"],
        })

    output = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("ESCALATION EVALUATION")
    print("=" * 60)

    print(f"Examples evaluated: {len(output)}")
    print()

    print("Decision distribution:")
    print(output["decision"].value_counts().to_string())
    print()

    escalation_rate = (
        output["decision"].eq("ESCALATE").mean()
    )

    print(f"Escalation rate: {escalation_rate:.2%}")
    print()

    print("Decision by intent:")
    print(
        pd.crosstab(
            output["golden_intent"],
            output["decision"],
        ).to_string()
    )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()