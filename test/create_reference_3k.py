import pandas as pd

REFERENCE_FILE = "results/amazonhelp_reference_pool.csv"
OUTPUT_FILE = "results/amazonhelp_reference_3k.csv"

SAMPLE_SIZE = 3000
RANDOM_STATE = 42


df = pd.read_csv(REFERENCE_FILE)

print("Original reference pool:", len(df))


# Remove very short/context-dependent messages
df = df[
    df["customer_message"].fillna("").str.len() > 30
].copy()

print("After removing short messages:", len(df))


# Random representative sample
sample = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=RANDOM_STATE
).reset_index(drop=True)


sample.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Created reference pool:", len(sample))
print("Saved:", OUTPUT_FILE)