
from matplotlib import text
import pandas as pd
from collections import defaultdict

FILE_PATH = "data/raw/twcs.csv"

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways"
]

# Start with 500,000 rows.
# We can increase this later if needed.
NROWS = 500000

print("Loading dataset...")

df = pd.read_csv(
    FILE_PATH,
    nrows=NROWS,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id"
    ]
)

# ---------------------------------------------------------
# Convert IDs
# ---------------------------------------------------------

df["tweet_id"] = pd.to_numeric(
    df["tweet_id"],
    errors="coerce"
)

df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"],
    errors="coerce"
)

df = df.dropna(subset=["tweet_id"])

# ---------------------------------------------------------
# Create tweet lookup
# ---------------------------------------------------------

tweets = df.set_index("tweet_id").to_dict("index")

# Parent -> child tweets
children = defaultdict(list)

for _, row in df.iterrows():

    tweet_id = row["tweet_id"]
    parent_id = row["in_response_to_tweet_id"]

    if pd.notna(parent_id) and parent_id in tweets:
        children[parent_id].append(tweet_id)

# ---------------------------------------------------------
# Find roots
# ---------------------------------------------------------

roots = []

for _, row in df.iterrows():

    tweet_id = row["tweet_id"]
    parent_id = row["in_response_to_tweet_id"]

    if pd.isna(parent_id) or parent_id not in tweets:
        roots.append(tweet_id)

# ---------------------------------------------------------
# Reconstruct conversations
# ---------------------------------------------------------

conversations = []

visited_global = set()

for root_id in roots:

    if root_id in visited_global:
        continue

    queue = [root_id]
    conversation_ids = []
    visited = set()

    while queue:

        current_id = queue.pop(0)

        if current_id in visited:
            continue

        if current_id not in tweets:
            continue

        visited.add(current_id)
        visited_global.add(current_id)

        conversation_ids.append(current_id)

        for child_id in children.get(current_id, []):
            queue.append(child_id)

    if len(conversation_ids) >= 2:

        messages = [
            tweets[tweet_id]
            for tweet_id in conversation_ids
        ]

        # Determine support brands involved
        support_accounts = set(
            message["author_id"]
            for message in messages
            if (
                message["inbound"] == False
                and message["author_id"] in BRANDS
            )
        )

        for brand in support_accounts:

            customer_count = sum(
                1
                for message in messages
                if message["inbound"] == True
            )

            support_count = sum(
                1
                for message in messages
                if (
                    message["inbound"] == False
                    and message["author_id"] == brand
                )
            )

            conversations.append({
                "brand": brand,
                "root_tweet_id": int(root_id),
                "message_count": len(messages),
                "customer_messages": customer_count,
                "support_messages": support_count
            })

# ---------------------------------------------------------
# Convert to DataFrame
# ---------------------------------------------------------

stats = pd.DataFrame(conversations)

print("\n" + "=" * 90)
print("BRAND CONVERSATION ANALYSIS")
print("=" * 90)

if stats.empty:

    print("No conversations found.")

else:

    summary = (
        stats
        .groupby("brand")
        .agg(
            conversations=("root_tweet_id", "count"),
            avg_messages=("message_count", "mean"),
            median_messages=("message_count", "median"),
            avg_customer_messages=("customer_messages", "mean"),
            avg_support_messages=("support_messages", "mean")
        )
        .sort_values(
            "conversations",
            ascending=False
        )
    )

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    summary.to_csv(
        "results/brand_conversation_summary.csv"
    )

    stats.to_csv(
        "results/brand_conversations.csv",
        index=False
    )

    print("\nResults saved:")
    print("  results/brand_conversation_summary.csv")
    print("  results/brand_conversations.csv")

print("\nDONE")
