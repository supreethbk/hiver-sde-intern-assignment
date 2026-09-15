"""Collect independent human quality ratings for a small calibration sample."""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("results/reply_generation_eval_20.csv")
OUTPUT_FILE = Path("evaluation/human_evaluation.csv")

SAMPLE_SIZE = 10


def main():
    df = pd.read_csv(INPUT_FILE)

    # Fixed sample for reproducibility.
    sample = df.sample(n=SAMPLE_SIZE, random_state=42)

    print("=" * 60)
    print("HUMAN EVALUATION")
    print("=" * 60)
    print()
    print("Rate each reply independently from 1 to 5:")
    print("1 = Very poor")
    print("2 = Poor")
    print("3 = Acceptable")
    print("4 = Good")
    print("5 = Excellent")
    print()
    print(f"You will rate {len(sample)} examples.")
    print()

    results = []

    for _, row in sample.iterrows():
        print("=" * 60)
        print(f"Example {row['example_id']}")
        print("=" * 60)

        print("\nCUSTOMER:")
        print(row["customer_message"])

        print("\nGENERATED REPLY:")
        print(row["generated_reply"])

        while True:
            score = input("\nYour overall score (1-5): ").strip()

            if score in {"1", "2", "3", "4", "5"}:
                score = int(score)
                break

            print("Please enter only 1, 2, 3, 4, or 5.")

        results.append({
            "example_id": row["example_id"],
            "human_score": score,
        })

        print()

    output = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("Human evaluation complete")
    print("=" * 60)
    print(f"Examples rated: {len(output)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()