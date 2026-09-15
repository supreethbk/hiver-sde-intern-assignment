"""Measure agreement between human and LLM judges."""

from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score


HUMAN_FILE = Path("evaluation/human_evaluation.csv")
LLM_FILE = Path("results/reply_quality_judge_20.csv")
OUTPUT_FILE = Path("results/judge_agreement.json")


def main():
    human = pd.read_csv(HUMAN_FILE)
    llm = pd.read_csv(LLM_FILE)

    merged = human.merge(
        llm[["example_id", "overall_score"]],
        on="example_id",
        how="inner",
    )

    if merged.empty:
        raise ValueError("No matching examples between human and LLM evaluations.")

    merged["human_score"] = merged["human_score"].astype(int)
    merged["overall_score"] = merged["overall_score"].astype(int)

    exact_agreement = (
        merged["human_score"] == merged["overall_score"]
    ).mean()

    weighted_kappa = cohen_kappa_score(
        merged["human_score"],
        merged["overall_score"],
        weights="quadratic",
    )

    result = {
        "examples_compared": int(len(merged)),
        "exact_agreement": round(float(exact_agreement), 4),
        "quadratic_weighted_kappa": round(float(weighted_kappa), 4),
        "score_scale": "Both human and LLM use direct 1-5 overall score",
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 50)
    print("LLM vs Human Agreement")
    print("=" * 50)
    print(f"Examples compared: {len(merged)}")
    print(f"Exact agreement: {exact_agreement:.2%}")
    print(f"Quadratic weighted kappa: {weighted_kappa:.3f}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()