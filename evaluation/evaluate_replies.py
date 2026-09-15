"""Evaluate generated replies."""
"""Evaluate generated replies with the LLM-as-judge."""

from pathlib import Path

import pandas as pd

from llm_judge import judge_reply

INPUT_FILE = Path("results/reply_generation_eval_20.csv")
OUTPUT_FILE = Path("results/reply_quality_judge_20.csv")


def main():
    df = pd.read_csv(INPUT_FILE)

    results = []

    print("Evaluating reply quality...")
    print(f"Examples: {len(df)}")

    for _, row in df.iterrows():
        print(f"Evaluating example {row['example_id']}...")

        try:
            judgment = judge_reply(
                row["customer_message"],
                row["generated_reply"],
            )

            results.append({
                "example_id": row["example_id"],
                "customer_message": row["customer_message"],
                "generated_reply": row["generated_reply"],
                **judgment,
            })

        except Exception as exc:
            results.append({
                "example_id": row["example_id"],
                "customer_message": row["customer_message"],
                "generated_reply": row["generated_reply"],
                "valid_judgment": False,
                "reason": str(exc),
            })

    results_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    valid_df = results_df[
        results_df["valid_judgment"] == True
    ].copy()

    print("\n========================================")
    print("Reply Quality Summary")
    print("========================================")

    print(f"Valid judgments: {len(valid_df)}/{len(results_df)}")

    if not valid_df.empty:
        criteria = [
            "relevance",
            "usefulness",
            "grounding",
            "factual_safety",
            "leakage_safety",
            "style",
        ]

        for criterion in criteria:
            print(
                f"{criterion}: "
                f"{valid_df[criterion].mean():.2f}/2"
            )

        print(
            f"\nAverage total score: "
            f"{valid_df['total_score'].mean():.2f}/12"
        )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()