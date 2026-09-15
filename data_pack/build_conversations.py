import pandas as pd

# Dataset location
FILE_PATH = "data/raw/twcs.csv"

# Read a manageable portion first
print("Loading dataset...")

df = pd.read_csv(
    FILE_PATH,
    nrows=50000,
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

# Convert IDs to numeric values
df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"],
    errors="coerce"
)

# Create lookup: tweet_id -> tweet row
tweets = df.set_index("tweet_id").to_dict("index")

print("\nDataset loaded:", len(df), "tweets")

# --------------------------------------------------
# Find tweets that have a parent tweet
# --------------------------------------------------

reply_tweets = df[
    df["in_response_to_tweet_id"].notna()
].head(10)

print("\n" + "=" * 80)
print("SAMPLE CONVERSATIONS")
print("=" * 80)

conversation_number = 1

for _, row in reply_tweets.iterrows():

    current_id = row["tweet_id"]

    # Follow the conversation backwards
    conversation = []

    while current_id in tweets:

        tweet = tweets[current_id]

        conversation.append(tweet)

        parent_id = tweet["in_response_to_tweet_id"]

        if pd.isna(parent_id):
            break

        current_id = parent_id

    # Reverse so oldest message appears first
    conversation.reverse()

    print(f"\n\nCONVERSATION {conversation_number}")
    print("-" * 80)

    for message in conversation:

        if message["inbound"]:
            speaker = "CUSTOMER"
        else:
            speaker = "SUPPORT"

        print(f"\n[{speaker}]")
        print(message["text"])

    conversation_number += 1

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
