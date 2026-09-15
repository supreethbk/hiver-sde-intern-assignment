import pandas as pd

INPUT_FILE = "results/amazonhelp_pairs_analyzed.csv"
OUTPUT_FILE = "results/intent_manual_sample.csv"

SAMPLE_SIZE = 200

print("Loading data...")

df = pd.read_csv(INPUT_FILE)

# Keep one row per customer message
df = df.drop_duplicates(
    subset=["customer_tweet_id"]
).copy()

# Remove empty messages
df["customer_message"] = (
    df["customer_message"]
    .fillna("")
    .map(lambda x: str(x).strip())
)

df = df[df["customer_message"] != ""]

print("Unique customer messages:", len(df))

# Random sample
sample = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
).copy()

# Keep only useful columns
columns = [
    "conversation_id",
    "customer_tweet_id",
    "customer_message",
    "support_response"
]

sample = sample[columns]

# Add empty column for our manual intent label
sample["intent"] = ""

# Add empty column for notes
sample["notes"] = ""

sample.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("SAMPLE CREATED")
print("========================================")

print("Sample size:", len(sample))
print("Saved to:", OUTPUT_FILE)

print("\nOpen this file in Excel.")
print("Read the customer message and support response.")
print("We will label these examples in the next step.")