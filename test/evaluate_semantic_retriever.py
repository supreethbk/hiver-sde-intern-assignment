import pickle
import requests
import pandas as pd
import numpy as np


GOLDEN_FILE = "results/amazonhelp_golden_labeled.csv"
CACHE_FILE = "results/amazonhelp_embeddings_3k.pkl"

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"

TOP_K = 5


# --------------------------------------------------
# Load data
# --------------------------------------------------

golden = pd.read_csv(GOLDEN_FILE)

with open(CACHE_FILE, "rb") as f:
    cache = pickle.load(f)

embeddings = np.array(
    cache["embeddings"],
    dtype=np.float32
)

customer_messages = cache["customer_messages"]


# --------------------------------------------------
# Query embedding
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
# Similarity
# --------------------------------------------------

def retrieve(query, top_k=TOP_K):

    query_embedding = get_embedding(query)

    scores = np.dot(
        embeddings,
        query_embedding
    ) / (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(query_embedding)
    )

    indices = np.argsort(scores)[-top_k:][::-1]

    return [
        (
            customer_messages[i],
            float(scores[i])
        )
        for i in indices
    ]


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

results = []

print("Evaluating semantic retrieval...")
print("Golden examples:", len(golden))
print("Reference examples:", len(embeddings))


for count, row in enumerate(
    golden.itertuples(index=False),
    1
):

    query = row.customer_message
    actual_intent = row.final_intent

    retrieved = retrieve(query)

    results.append({
        "example_id": row.example_id,
        "customer_message": query,
        "actual_intent": actual_intent,
        "top_similarity": retrieved[0][1],
        "top_match": retrieved[0][0]
    })

    if count % 20 == 0:
        print(f"Processed {count}/{len(golden)}")


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

OUTPUT_FILE = "results/semantic_retrieval_results_200.csv"

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nDone.")
print("Saved:", OUTPUT_FILE)