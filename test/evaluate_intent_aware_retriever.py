import pickle
import requests
import pandas as pd
import numpy as np


GOLDEN_FILE = "results/amazonhelp_golden_labeled.csv"
CACHE_FILE = "results/amazonhelp_embeddings_3k.pkl"
LABELED_REFERENCE_FILE = "results/amazonhelp_reference_3k_labeled.csv"

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"

TOP_K = 5


# --------------------------------------------------
# Load data
# --------------------------------------------------

golden = pd.read_csv(GOLDEN_FILE)
reference = pd.read_csv(LABELED_REFERENCE_FILE)

with open(CACHE_FILE, "rb") as f:
    cache = pickle.load(f)

embeddings = np.array(
    cache["embeddings"],
    dtype=np.float32
)

customer_messages = cache["customer_messages"]

reference["intent"] = reference["intent"].fillna("OTHER")


# --------------------------------------------------
# Embed query
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

def get_scores(query_embedding):

    return np.dot(
        embeddings,
        query_embedding
    ) / (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(query_embedding)
    )


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

results = []

print("Evaluating intent-aware retrieval...")
print("Golden examples:", len(golden))
print("Reference examples:", len(reference))


for count, row in enumerate(
    golden.itertuples(index=False),
    1
):

    query = row.customer_message
    actual_intent = row.final_intent

    query_embedding = get_embedding(query)

    scores = get_scores(query_embedding)

    # ----------------------------------------------
    # Predict intent using existing local classifier
    # ----------------------------------------------

    from classifier import classify

    predicted_intent = classify(query)

    # ----------------------------------------------
    # Only retrieve examples with predicted intent
    # ----------------------------------------------

    mask = (
        reference["intent"].values
        == predicted_intent
    )

    candidate_indices = np.where(mask)[0]

    if len(candidate_indices) == 0:

        candidate_indices = np.arange(
            len(reference)
        )

    candidate_scores = scores[candidate_indices]

    top_positions = np.argsort(
        candidate_scores
    )[-TOP_K:][::-1]

    top_indices = candidate_indices[
        top_positions
    ]

    retrieved_intents = [
        reference.iloc[i]["intent"]
        for i in top_indices
    ]

    top_match_intent = retrieved_intents[0]

    intent_match = (
        top_match_intent == actual_intent
    )

    results.append({
        "example_id": row.example_id,
        "customer_message": query,
        "actual_intent": actual_intent,
        "predicted_intent": predicted_intent,
        "top_match_intent": top_match_intent,
        "intent_match": intent_match,
        "top_similarity": float(
            scores[top_indices[0]]
        )
    })

    if count % 20 == 0:

        print(
            f"Processed {count}/{len(golden)}"
        )


# --------------------------------------------------
# Save
# --------------------------------------------------

results_df = pd.DataFrame(results)

OUTPUT_FILE = (
    "results/"
    "intent_aware_retrieval_results_200.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = (
    results_df["intent_match"].mean()
    * 100
)


print("\n================================")
print("Intent-aware retrieval complete")
print("================================")

print(
    f"Top-1 intent match: "
    f"{accuracy:.2f}%"
)

print(
    "Correct:",
    results_df["intent_match"].sum(),
    "/",
    len(results_df)
)

print(
    "Saved:",
    OUTPUT_FILE
)