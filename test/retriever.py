import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REFERENCE_FILE = "results/amazonhelp_reference_pool.csv"


# Load historical customer → support pairs
df = pd.read_csv(REFERENCE_FILE)

customer_messages = df["customer_message"].fillna("").tolist()

# Build TF-IDF index
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

customer_vectors = vectorizer.fit_transform(customer_messages)


def retrieve(query, top_k=3):
    """
    Retrieve the most similar historical customer messages
    and their corresponding Amazon support responses.
    """

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        customer_vectors
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "customer_message": df.iloc[index]["customer_message"],
            "support_response": df.iloc[index]["support_response"],
            "similarity": round(float(similarities[index]), 4)
        })

    return results


if __name__ == "__main__":

    query = "Where is my Amazon package?"

    results = retrieve(query, top_k=3)

    print("\nQuery:")
    print(query)

    print("\nRetrieved historical conversations:")

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print("Similarity:", result["similarity"])
        print("Customer:", result["customer_message"])
        print("Amazon:", result["support_response"])