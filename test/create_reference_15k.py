import pandas as pd

REFERENCE_FILE = "results/amazonhelp_reference_pool.csv"
OUTPUT_FILE = "results/amazonhelp_reference_15k.csv"

SAMPLE_SIZE = 15000
RANDOM_STATE = 42


# Load the already leakage-safe reference pool
df = pd.read_csv(REFERENCE_FILE)

print("Original reference pool:", len(df))


# Random representative sample
sample = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=RANDOM_STATE
).reset_index(drop=True)


# Save
sample.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Created reference pool:", len(sample))
print("Saved:", OUTPUT_FILE)