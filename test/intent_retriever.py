import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fewshot_classifier_v2 import classify


REFERENCE_FILE = "results/reference_labeled_100.csv"

df = pd.read_csv(REFERENCE_FILE)

df["customer_message"] = df["customer_message"].fillna("")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

vectors = vectorizer.fit_transform(df["customer_message"])


def retrieve(query, top_k=3):

    # Step 1: classify the incoming message
    predicted_intent = classify(query)

    # Step 2: keep only reference examples with that intent
    intent_df = df[df["intent"] == predicted_intent].copy()

    if len(intent_df) == 0:
        return predicted_intent, []

    # Step 3: calculate similarity only inside that intent
    intent_vectors = vectorizer.transform(
        intent_df["customer_message"]
    )

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        intent_vectors
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        row = intent_df.iloc[index]

        results.append({
            "customer_message": row["customer_message"],
            "support_response": row["support_response"],
            "similarity": round(float(similarities[index]), 4),
            "intent": row["intent"]
        })

    return predicted_intent, results


if __name__ == "__main__":

    query = "Where is my Amazon package?"

    intent, results = retrieve(query, top_k=3)

    print("\nPredicted intent:")
    print(intent)

    print("\nRetrieved examples:")

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print("Intent:", result["intent"])
        print("Similarity:", result["similarity"])
        print("Customer:", result["customer_message"])
        print("Amazon:", result["support_response"])