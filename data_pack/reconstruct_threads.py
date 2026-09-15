import pandas as pd
from collections import defaultdict

FILE_PATH = "data/raw/twcs.csv"

# We test on 100,000 rows first.
# We will NOT process the full dataset until this works.
NROWS = 100000

print("Loading tweets...")

df = pd.read_csv(
    FILE_PATH,
    nrows=NROWS,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
)

# Convert IDs to numbers
df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"],
    errors="coerce"
)

# Remove invalid tweet IDs
df = df.dropna(subset=["tweet_id"])

# ---------------------------------------------------------
# Create lookup tables
# ---------------------------------------------------------

tweets = df.set_index("tweet_id").to_dict("index")

# parent tweet -> children tweets
children = defaultdict(list)

for _, row in df.iterrows():

    tweet_id = row["tweet_id"]
    parent_id = row["in_response_to_tweet_id"]

    if pd.notna(parent_id):
        children[parent_id].append(tweet_id)

# ---------------------------------------------------------
# Find conversation roots
# ---------------------------------------------------------

roots = []

for _, row in df.iterrows():

    tweet_id = row["tweet_id"]
    parent_id = row["in_response_to_tweet_id"]

    # No parent = possible conversation root
    if pd.isna(parent_id) or parent_id not in tweets:
        roots.append(tweet_id)

print("\nTweets loaded:", len(df))
print("Conversation roots found:", len(roots))

# ---------------------------------------------------------
# Build conversations
# ---------------------------------------------------------

conversations = []

for root_id in roots:

    queue = [root_id]
    visited = set()
    conversation_ids = []

    while queue:

        current_id = queue.pop(0)

        if current_id in visited:
            continue

        if current_id not in tweets:
            continue

        visited.add(current_id)
        conversation_ids.append(current_id)

        # Follow replies
        for child_id in children.get(current_id, []):
            queue.append(child_id)

    # Keep conversations with at least 2 messages
    if len(conversation_ids) >= 2:

        messages = []

        for tweet_id in conversation_ids:

            tweet = tweets[tweet_id]

            speaker = (
                "CUSTOMER"
                if tweet["inbound"]
                else "SUPPORT"
            )

            messages.append({
                "tweet_id": int(tweet_id),
                "speaker": speaker,
                "author_id": tweet["author_id"],
                "created_at": tweet["created_at"],
                "text": tweet["text"]
            })

        # Sort chronologically
        messages.sort(
            key=lambda x: x["created_at"]
        )

        conversations.append({
            "root_tweet_id": int(root_id),
            "message_count": len(messages),
            "messages": messages
        })

# ---------------------------------------------------------
# Display first 5 unique conversations
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("FIRST 5 RECONSTRUCTED CONVERSATIONS")
print("=" * 80)

for number, conversation in enumerate(
    conversations[:5],
    start=1
):

    print(f"\nCONVERSATION {number}")
    print("-" * 80)

    print(
        "Root tweet:",
        conversation["root_tweet_id"]
    )

    print(
        "Messages:",
        conversation["message_count"]
    )

    for message in conversation["messages"]:

        print(
            f"\n[{message['speaker']}] "
            f"(tweet {message['tweet_id']})"
        )

        print(message["text"])

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
