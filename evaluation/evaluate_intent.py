"""Evaluate intent classification results."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


RESULTS_FILE = Path("results/classification_results_200.csv")


def main():
    df = pd.read_csv(RESULTS_FILE)

    # Support the column names already used by the classifier evaluation.
    gold_col = "actual_intent"
    pred_col = "predicted_intent"

    if gold_col not in df.columns or pred_col not in df.columns:
        raise ValueError(
            f"Expected columns '{gold_col}' and '{pred_col}'. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=[gold_col, pred_col])

    accuracy = accuracy_score(
        df[gold_col],
        df[pred_col],
    )

    print("=" * 60)
    print("INTENT CLASSIFICATION EVALUATION")
    print("=" * 60)
    print(f"Examples evaluated: {len(df)}")
    print(f"Accuracy: {accuracy:.2%}")
    print()

    print("Per-intent results:")
    print(classification_report(
            df[gold_col],
            df[pred_col],
            labels=sorted(df[gold_col].unique()),
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    labels = sorted(df[gold_col].unique())

    matrix = confusion_matrix(
        df[gold_col],
        df[pred_col],
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    print(matrix_df.to_string())
    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()