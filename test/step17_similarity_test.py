import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GOLDEN_FILE = "results/amazonhelp_golden_labeled.csv"
REFERENCE_FILE = "results/reference_labeled_100.csv"

golden = pd.read_csv(GOLDEN_FILE)
reference = pd.read_csv(REFERENCE_FILE)

# Fill missing text safely
golden_text = golden["customer_message"].fillna("")
reference_text = reference["customer_message"].fillna("")

# TF-IDF converts messages into numerical vectors
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1
)

all_text = pd.concat([golden_text, reference_text])

vectorizer.fit(all_text)

golden_vectors = vectorizer.transform(golden_text)
reference_vectors = vectorizer.transform(reference_text)

# Similarity between every Golden message and every reference message
similarities = cosine_similarity(golden_vectors, reference_vectors)

results = []

for i in range(len(golden)):
    # Find most similar reference example
    best_index = similarities[i].argmax()
    best_score = similarities[i][best_index]

    results.append({
        "example_id": golden.iloc[i]["example_id"],
        "customer_message": golden.iloc[i]["customer_message"],
        "actual_intent": golden.iloc[i]["final_intent"],
        "nearest_reference_intent": reference.iloc[best_index]["intent"],
        "similarity_score": round(float(best_score), 4),
        "nearest_reference_message": reference.iloc[best_index]["customer_message"]
    })

results_df = pd.DataFrame(results)

output_file = "results/baseline2_similarity_results.csv"
results_df.to_csv(output_file, index=False)

# Accuracy of nearest-neighbor intent
accuracy = (
    results_df["actual_intent"]
    == results_df["nearest_reference_intent"]
).mean()

print("Baseline 2 - TF-IDF Nearest Neighbor")
print("-------------------------------------")
print(f"Golden examples: {len(results_df)}")
print(f"Correct: {(results_df['actual_intent'] == results_df['nearest_reference_intent']).sum()}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Results saved to: {output_file}")

print("\nSample predictions:")
print(
    results_df[
        [
            "example_id",
            "actual_intent",
            "nearest_reference_intent",
            "similarity_score"
        ]
    ].head(10).to_string(index=False)
)