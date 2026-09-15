import pandas as pd
from collections import defaultdict, deque

FILE_PATH = "data/raw/twcs.csv"

BRAND = "AmazonHelp"

NROWS = 500000
SAMPLE_SIZE = 100

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
# Clean IDs
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
# Tweet lookup
# ---------------------------------------------------------

tweets = df.set_index("tweet_id").to_dict("index")

# ---------------------------------------------------------
# Parent -> children
# ---------------------------------------------------------

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

amazon_conversations = []

visited_global = set()

for root_id in roots:

    if root_id in visited_global:
        continue

    queue = deque([root_id])
    conversation_ids = []
    visited = set()

    while queue:

        current_id = queue.popleft()

        if current_id in visited:
            continue

        if current_id not in tweets:
            continue

        visited.add(current_id)
        visited_global.add(current_id)

        conversation_ids.append(current_id)

        for child_id in children.get(current_id, []):
            queue.append(child_id)

    if len(conversation_ids) < 2:
        continue

    messages = []

    for tweet_id in conversation_ids:

        message = tweets[tweet_id].copy()
        message["tweet_id"] = tweet_id

        messages.append(message)

    # -----------------------------------------------------
    # Keep only conversations involving AmazonHelp
    # -----------------------------------------------------

    amazon_support_present = any(
        message["author_id"] == BRAND
        and message["inbound"] == False
        for message in messages
    )

    if not amazon_support_present:
        continue

    # -----------------------------------------------------
    # Keep normal-sized conversations
    # -----------------------------------------------------

    if len(messages) > 20:
        continue

    amazon_conversations.append(messages)

# ---------------------------------------------------------
# Extract CUSTOMER messages from Amazon conversations
# ---------------------------------------------------------

customer_messages = []

for conversation in amazon_conversations:

    for message in conversation:

        if (
            message["inbound"] == True
            and pd.notna(message["text"])
        ):

            text = str(message["text"]).strip()

            if text:
                customer_messages.append({
                    "tweet_id": int(message["tweet_id"]),
                    "created_at": message["created_at"],
                    "text": text
                })

# ---------------------------------------------------------
# Remove duplicates
# ---------------------------------------------------------

customer_df = pd.DataFrame(customer_messages)

if customer_df.empty:

    print("No AmazonHelp customer messages found.")

    raise SystemExit

customer_df = customer_df.drop_duplicates(
    subset=["tweet_id"]
)

# ---------------------------------------------------------
# Sample
# ---------------------------------------------------------

sample_size = min(
    SAMPLE_SIZE,
    len(customer_df)
)

sample = customer_df.sample(
    n=sample_size,
    random_state=42
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_file = (
    "data/brand_samples/"
    "AmazonHelp_customer_intent_sample.csv"
)

sample.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 100)
print("AMAZONHELP CUSTOMER INTENT SAMPLE")
print("=" * 100)

print(
    f"\nAmazonHelp conversations found: "
    f"{len(amazon_conversations)}"
)

print(
    f"AmazonHelp customer messages available: "
    f"{len(customer_df)}"
)

print(
    f"Sample size: {len(sample)}"
)

# ---------------------------------------------------------
# Print sample
# ---------------------------------------------------------

for i, (_, row) in enumerate(
    sample.iterrows(),
    start=1
):

    print("\n" + "-" * 100)

    print(f"[{i}] Tweet ID: {row['tweet_id']}")
    print(f"Time: {row['created_at']}")
    print(f"Text: {row['text']}")

print("\n" + "=" * 100)
print("INTENT DISCOVERY SAMPLE COMPLETE")
print("=" * 100)
