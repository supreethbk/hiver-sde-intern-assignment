"""Trivial baseline implementation."""
"""Trivial majority-class baseline for intent classification."""

from pathlib import Path
import pandas as pd


GOLDEN_PATH = Path("data/golden/golden_set.csv")


def main():
    df = pd.read_csv(GOLDEN_PATH)

    labels = df["final_intent"].dropna()

    majority_intent = labels.value_counts().index[0]
    correct = (labels == majority_intent).sum()
    total = len(labels)
    accuracy = correct / total

    print("Trivial Baseline: Majority Intent")
    print(f"Golden examples: {total}")
    print(f"Majority intent: {majority_intent}")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()