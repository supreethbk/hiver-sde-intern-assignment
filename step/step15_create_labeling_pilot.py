import pandas as pd

INPUT_FILE = "results/amazonhelp_reference_pool.csv"
OUTPUT_FILE = "results/reference_labeling_pilot_100.csv"

df = pd.read_csv(INPUT_FILE)

# Reproducible random sample
sample = df.sample(n=100, random_state=42).copy()

# Keep only the fields needed for labeling
sample = sample[
    [
        "conversation_id",
        "customer_tweet_id",
        "support_tweet_id",
        "customer_message",
        "support_response",
    ]
]

sample.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("REFERENCE LABELING PILOT CREATED")
print("=" * 60)
print("Reference pool:", len(df))
print("Pilot examples:", len(sample))
print("Saved to:", OUTPUT_FILE)
