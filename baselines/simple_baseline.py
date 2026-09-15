"""Simple baseline implementation."""
"""Simple TF-IDF nearest-neighbor baseline for intent classification."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


GOLDEN_FILE = Path("data/golden/golden_set.csv")
REFERENCE_FILE = Path("results/reference_labeled_100.csv")
OUTPUT_FILE = Path("results/baseline2_similarity_results.csv")


def main():
    golden = pd.read_csv(GOLDEN_FILE)
    reference = pd.read_csv(REFERENCE_FILE)

    golden_text = golden["customer_message"].fillna("")
    reference_text = reference["customer_message"].fillna("")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
    )

    vectorizer.fit(pd.concat([golden_text, reference_text]))

    golden_vectors = vectorizer.transform(golden_text)
    reference_vectors = vectorizer.transform(reference_text)

    similarities = cosine_similarity(golden_vectors, reference_vectors)

    results = []

    for i in range(len(golden)):
        best_index = similarities[i].argmax()
        best_score = similarities[i][best_index]

        results.append({
            "example_id": golden.iloc[i]["example_id"],
            "customer_message": golden.iloc[i]["customer_message"],
            "actual_intent": golden.iloc[i]["final_intent"],
            "nearest_reference_intent": reference.iloc[best_index]["intent"],
            "similarity_score": round(float(best_score), 4),
            "nearest_reference_message":
                reference.iloc[best_index]["customer_message"],
        })

    results_df = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    correct = (
        results_df["actual_intent"]
        == results_df["nearest_reference_intent"]
    ).sum()

    accuracy = correct / len(results_df)

    print("Simple Baseline - TF-IDF Nearest Neighbor")
    print("------------------------------------------")
    print(f"Golden examples: {len(results_df)}")
    print(f"Correct: {correct}/{len(results_df)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()