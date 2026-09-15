import pandas as pd

HISTORICAL_FILE = "results/amazonhelp_pairs_analyzed.csv"
GOLDEN_FILE = "results/amazonhelp_golden_labeled.csv"
OUTPUT_FILE = "results/amazonhelp_reference_pool.csv"

historical = pd.read_csv(HISTORICAL_FILE)
golden = pd.read_csv(GOLDEN_FILE)

# Remove every historical row whose customer message
# appears in the 200-example Golden Set.
golden_messages = set(golden["customer_message"].dropna())

reference = historical[
    ~historical["customer_message"].isin(golden_messages)
].copy()

reference.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("LEAKAGE-FREE REFERENCE POOL CREATED")
print("=" * 60)

print("Historical pairs:", len(historical))
print("Golden examples:", len(golden))
print("Reference pool:", len(reference))

print("\nGolden messages remaining in reference pool:",
      reference["customer_message"].isin(golden_messages).sum())

print("\nSaved to:", OUTPUT_FILE)