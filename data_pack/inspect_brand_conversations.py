import pandas as pd
from collections import defaultdict, deque

FILE_PATH = "data/raw/twcs.csv"

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support"
]

NROWS = 500000
CONVERSATIONS_PER_BRAND = 5

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
# Create tweet lookup
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

conversations = []

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

    # Get complete message records
    messages = []

    for tweet_id in conversation_ids:

        message = tweets[tweet_id].copy()

        # Add tweet_id back into the dictionary
        message["tweet_id"] = tweet_id

        messages.append(message)

    # -----------------------------------------------------
    # Determine support brands
    # -----------------------------------------------------

    support_accounts = {
        message["author_id"]
        for message in messages
        if (
            message["inbound"] == False
            and message["author_id"] in BRANDS
        )
    }

    # -----------------------------------------------------
    # Store conversation
    # -----------------------------------------------------

    for brand in support_accounts:

        conversations.append({
            "brand": brand,
            "root_tweet_id": int(root_id),
            "messages": messages
        })

# ---------------------------------------------------------
# Print sample conversations
# ---------------------------------------------------------

print("\n" + "=" * 100)
print("SAMPLE BRAND CONVERSATIONS")
print("=" * 100)

for brand in BRANDS:

    brand_conversations = [
        c for c in conversations
        if c["brand"] == brand
    ]

    print("\n\n")
    print("#" * 100)
    print(f"BRAND: {brand}")
    print("#" * 100)

    if not brand_conversations:
        print("No conversations found.")
        continue

    # -----------------------------------------------------
    # Remove extremely large conversations for inspection
    # -----------------------------------------------------

    normal_conversations = [
        c for c in brand_conversations
        if 2 <= len(c["messages"]) <= 20
    ]

    # Prefer normal-sized conversations
    if len(normal_conversations) >= CONVERSATIONS_PER_BRAND:
        selected = normal_conversations[:CONVERSATIONS_PER_BRAND]
    else:
        selected = brand_conversations[:CONVERSATIONS_PER_BRAND]

    # -----------------------------------------------------
    # Print conversations
    # -----------------------------------------------------

    for i, conversation in enumerate(selected, start=1):

        print("\n" + "-" * 100)

        print(
            f"Conversation {i} | "
            f"Root Tweet: {conversation['root_tweet_id']} | "
            f"Messages: {len(conversation['messages'])}"
        )

        print("-" * 100)

        messages = sorted(
            conversation["messages"],
            key=lambda x: x["created_at"]
        )

        for j, message in enumerate(messages, start=1):

            if message["inbound"]:
                speaker = "CUSTOMER"
            else:
                speaker = "SUPPORT"

            print(f"\n[{j}] {speaker}")
            print(f"Time: {message['created_at']}")
            print(f"Author: {message['author_id']}")
            print(f"Tweet ID: {int(message['tweet_id'])}")
            print(f"Text: {message['text']}")

print("\n" + "=" * 100)
print("INSPECTION COMPLETE")
print("=" * 100)