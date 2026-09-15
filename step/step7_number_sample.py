import pandas as pd

INPUT_FILE = "results/intent_manual_sample.csv"
OUTPUT_FILE = "results/intent_manual_review.csv"

df = pd.read_csv(INPUT_FILE)

# Add a simple example number
df.insert(0, "example_id", range(1, len(df) + 1))

# Keep the columns we need
columns = [
    "example_id",
    "customer_message",
    "support_response",
    "intent",
    "notes"
]

df = df[columns]

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("========================================")
print("REVIEW FILE CREATED")
print("========================================")
print("Examples:", len(df))
print("Saved to:", OUTPUT_FILE)

print("ex 20-30 examples:\n")

for _, row in df.iloc[179:200].iterrows():

    print("----------------------------------------")
    print("Example:", row["example_id"])
    print("Customer:", row["customer_message"])
    print("Support :", row["support_response"])