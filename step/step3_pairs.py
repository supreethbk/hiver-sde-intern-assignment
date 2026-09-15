import pandas as pd

INPUT_FILE = "results/amazonhelp_conversations.csv"

print("Loading AmazonHelp dataset...")

df = pd.read_csv(INPUT_FILE)

# Make IDs strings so they can be compared safely
df["tweet_id"] = df["tweet_id"].astype(str)

df["in_response_to_tweet_id"] = (
    pd.to_numeric(
        df["in_response_to_tweet_id"],
        errors="coerce"
    )
)


print("\n========================================")
print("DATASET")
print("========================================")

print("Total rows:", len(df))
print("Conversations:", df["conversation_id"].nunique())


# ------------------------------------------------------------
# Create a lookup:
# tweet_id -> tweet information
# ------------------------------------------------------------

tweet_lookup = df.set_index("tweet_id")


# ------------------------------------------------------------
# Find AmazonHelp replies to customer messages
# ------------------------------------------------------------

pairs = []

for _, row in df[df["speaker"] == "support"].iterrows():

    parent_id = row["in_response_to_tweet_id"]

    if pd.isna(parent_id):
        continue

    parent_id = str(int(parent_id))

    if parent_id not in tweet_lookup.index:
        continue

    parent = tweet_lookup.loc[parent_id]

    # We only want:
    #
    # CUSTOMER
    #     ↓
    # AMAZONHELP
    #
    if parent["speaker"] != "customer":
        continue

    customer_text = str(parent["text"]).strip()
    support_text = str(row["text"]).strip()

    if not customer_text or not support_text:
        continue

    pairs.append({
        "conversation_id": row["conversation_id"],
        "customer_tweet_id": parent_id,
        "support_tweet_id": row["tweet_id"],
        "customer_message": customer_text,
        "support_response": support_text,
        "customer_created_at": parent["created_at"],
        "support_created_at": row["created_at"]
    })


pairs_df = pd.DataFrame(pairs)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n========================================")
print("CUSTOMER → SUPPORT PAIRS")
print("========================================")

print("Pairs found:", len(pairs_df))

if len(pairs_df) > 0:

    print(
        "Unique conversations containing pairs:",
        pairs_df["conversation_id"].nunique()
    )

    print(
        "Unique customer messages:",
        pairs_df["customer_tweet_id"].nunique()
    )

    print(
        "Unique support responses:",
        pairs_df["support_tweet_id"].nunique()
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT_FILE = "results/amazonhelp_customer_support_pairs.csv"

pairs_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)


# ------------------------------------------------------------
# Show examples
# ------------------------------------------------------------

print("\n========================================")
print("EXAMPLES")
print("========================================")

for i, row in pairs_df.head(10).iterrows():

    print("\n----------------------------------------")

    print("Customer:")
    print(row["customer_message"])

    print("\nAmazonHelp:")
    print(row["support_response"])


print("\nDONE!")
