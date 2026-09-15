import requests
import pandas as pd
import numpy as np
import pickle
import os
import time


REFERENCE_FILE = "results/amazonhelp_reference_3k.csv"
CACHE_FILE = "results/amazonhelp_embeddings_3k.pkl"
CHECKPOINT_FILE = "results/embedding_checkpoint.pkl"

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"

BATCH_SIZE = 100


# --------------------------------------------------
# Load reference data
# --------------------------------------------------

df = pd.read_csv(REFERENCE_FILE)
df["customer_message"] = df["customer_message"].fillna("")

texts = df["customer_message"].tolist()
support_responses = df["support_response"].fillna("").tolist()

total = len(texts)

print(f"Total messages: {total}")
print(f"Batch size: {BATCH_SIZE}")


# --------------------------------------------------
# Resume from checkpoint if available
# --------------------------------------------------

if os.path.exists(CHECKPOINT_FILE):

    print("Checkpoint found. Resuming...")

    with open(CHECKPOINT_FILE, "rb") as f:
        checkpoint = pickle.load(f)

    all_embeddings = checkpoint["embeddings"]
    start_index = checkpoint["processed"]

    print(f"Already processed: {start_index}/{total}")

else:

    all_embeddings = []
    start_index = 0


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

start_time = time.time()

for start in range(start_index, total, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total)

    batch = texts[start:end]

    try:

        for attempt in range(5):

            try:

                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL,
                        "input": batch
                    },
                    timeout=300
                )

                if response.status_code == 200:
                    break

                print(
                    f"Batch {start}-{end} failed "
                    f"(attempt {attempt + 1}/5): "
                    f"{response.text[:200]}"
                )

                time.sleep(2)

            except requests.RequestException as e:

                print(
                    f"Connection error "
                    f"(attempt {attempt + 1}/5): {e}"
                )

                time.sleep(2)

        else:
            raise RuntimeError(
                f"Failed after 5 attempts: {start}-{end}"
            )


        data = response.json()

        embeddings = data["embeddings"]

        all_embeddings.extend(embeddings)

        processed = end

        elapsed = time.time() - start_time

        rate = (processed - start_index) / elapsed if elapsed > 0 else 0

        print(
            f"Processed {processed}/{total} "
            f"({processed / total * 100:.1f}%) "
            f"| {rate:.1f} msg/sec"
        )


        # ------------------------------------------
        # Save checkpoint every batch
        # ------------------------------------------

        with open(CHECKPOINT_FILE, "wb") as f:

            pickle.dump(
                {
                    "embeddings": all_embeddings,
                    "processed": processed
                },
                f
            )


    except Exception as e:

        print("\nERROR:")
        print(e)

        print("\nProgress has been saved.")
        print("Run the script again to resume.")

        raise


# --------------------------------------------------
# Convert to NumPy array
# --------------------------------------------------

all_embeddings = np.array(
    all_embeddings,
    dtype=np.float32
)


# --------------------------------------------------
# Save final cache
# --------------------------------------------------

with open(CACHE_FILE, "wb") as f:

    pickle.dump(
        {
            "embeddings": all_embeddings,
            "customer_messages": texts,
            "support_responses": support_responses
        },
        f
    )


# --------------------------------------------------
# Remove checkpoint
# --------------------------------------------------

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)


elapsed = time.time() - start_time


print("\n================================")
print("Embedding generation complete")
print("================================")

print("Embedding shape:", all_embeddings.shape)

print(
    f"Total time: {elapsed / 60:.2f} minutes"
)

print(
    "Saved:",
    CACHE_FILE
)