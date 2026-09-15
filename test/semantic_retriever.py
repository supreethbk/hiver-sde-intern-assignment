import requests
import pickle
import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CACHE_FILE = "results/amazonhelp_embeddings_3k.pkl"

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"


# --------------------------------------------------
# Load precomputed embeddings
# --------------------------------------------------

with open(CACHE_FILE, "rb") as f:
    data = pickle.load(f)


embeddings = np.array(
    data["embeddings"],
    dtype=np.float32
)

customer_messages = data["customer_messages"]
support_responses = data["support_responses"]


print(f"Loaded embeddings: {embeddings.shape[0]}")
print(f"Embedding dimensions: {embeddings.shape[1]}")


# --------------------------------------------------
# Embed the new query
# --------------------------------------------------

def get_embedding(text):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "input": [text]
        },
        timeout=60
    )

    response.raise_for_status()

    return np.array(
        response.json()["embeddings"][0],
        dtype=np.float32
    )


# --------------------------------------------------
# Semantic retrieval
# --------------------------------------------------

def retrieve(query, top_k=5):

    query_embedding = get_embedding(query)

    # Normalize query
    query_norm = np.linalg.norm(query_embedding)

    # Normalize all historical embeddings
    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    # Cosine similarity
    scores = np.dot(
        embeddings,
        query_embedding
    ) / (
        embedding_norms * query_norm
    )

    # Get highest scores
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "customer_message": customer_messages[index],
            "support_response": support_responses[index],
            "similarity": round(
                float(scores[index]),
                4
            )
        })

    return results


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    query = "Where is my Amazon package?"

    print("\nQuery:")
    print(query)

    print("\nSemantic retrieval results:")

    results = retrieve(
        query,
        top_k=5
    )

    for i, result in enumerate(results, 1):

        print(f"\n--- Result {i} ---")

        print(
            "Similarity:",
            result["similarity"]
        )

        print(
            "Customer:",
            result["customer_message"]
        )

        print(
            "Amazon:",
            result["support_response"]
        )