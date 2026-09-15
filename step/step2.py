import pandas as pd
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/raw/twcs.csv"

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "amazonhelp_conversations.csv"


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

print("Loading Twitter Customer Support dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total tweets: {len(df):,}")


# ============================================================
# STEP 2 — PREPARE COLUMNS
# ============================================================

df["tweet_id"] = df["tweet_id"].astype(str)

df["author_id"] = df["author_id"].astype(str)

df["in_response_to_tweet_id"] = (
    pd.to_numeric(
        df["in_response_to_tweet_id"],
        errors="coerce"
    )
)


# ============================================================
# STEP 3 — IDENTIFY AMAZONHELP TWEETS
# ============================================================

amazon_tweets = df[
    df["author_id"].str.lower() == "amazonhelp"
].copy()

print(f"AmazonHelp tweets: {len(amazon_tweets):,}")


# ============================================================
# STEP 4 — FIND CONVERSATIONS INVOLVING AMAZONHELP
# ============================================================

# A conversation can contain:
#   Customer → AmazonHelp
#   AmazonHelp → Customer
#   Customer → Customer
#
# We identify tweets directly connected to AmazonHelp
# through the response relationship.

amazon_ids = set(amazon_tweets["tweet_id"])

connected = df[
    (
        df["tweet_id"].isin(amazon_ids)
    )
    |
    (
        df["in_response_to_tweet_id"].astype("Int64")
        .astype(str)
        .isin(amazon_ids)
    )
].copy()


# ============================================================
# STEP 5 — CREATE A CONVERSATION ROOT
# ============================================================

tweet_parent = dict(
    zip(
        df["tweet_id"],
        df["in_response_to_tweet_id"]
    )
)


def find_root(tweet_id):
    """
    Follow parent tweets until reaching the
    beginning of the conversation.
    """

    visited = set()
    current = str(tweet_id)

    while current in tweet_parent:

        if current in visited:
            break

        visited.add(current)

        parent = tweet_parent[current]

        if pd.isna(parent):
            break

        parent = str(int(parent))

        if parent not in tweet_parent:
            break

        current = parent

    return current


print("Finding conversation roots...")

connected["conversation_id"] = connected["tweet_id"].apply(find_root)


# ============================================================
# STEP 6 — KEEP CONVERSATIONS THAT CONTAIN AMAZONHELP
# ============================================================

amazon_conversation_ids = set(
    connected.loc[
        connected["author_id"].str.lower() == "amazonhelp",
        "conversation_id"
    ]
)

conversations = connected[
    connected["conversation_id"].isin(amazon_conversation_ids)
].copy()


# ============================================================
# STEP 7 — SORT MESSAGES CHRONOLOGICALLY
# ============================================================

conversations["created_at"] = pd.to_datetime(
    conversations["created_at"],
    errors="coerce"
)

conversations = conversations.sort_values(
    ["conversation_id", "created_at"]
)


# ============================================================
# STEP 8 — ADD SPEAKER TYPE
# ============================================================

conversations["speaker"] = conversations["author_id"].apply(
    lambda x: "support"
    if str(x).lower() == "amazonhelp"
    else "customer"
)


# ============================================================
# STEP 9 — SAVE
# ============================================================

columns = [
    "conversation_id",
    "tweet_id",
    "created_at",
    "author_id",
    "speaker",
    "text",
    "in_response_to_tweet_id",
    "response_tweet_id",
]

conversations[columns].to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# STEP 10 — SUMMARY
# ============================================================

print("\n========================================")
print("AMAZONHELP DATASET CREATED")
print("========================================")

print(
    f"Conversations: "
    f"{conversations['conversation_id'].nunique():,}"
)

print(
    f"Tweets: "
    f"{len(conversations):,}"
)

print(
    f"Customer messages: "
    f"{(conversations['speaker'] == 'customer').sum():,}"
)

print(
    f"AmazonHelp messages: "
    f"{(conversations['speaker'] == 'support').sum():,}"
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("\nFirst few rows:")
print(
    conversations[
        [
            "conversation_id",
            "speaker",
            "text"
        ]
    ].head(20).to_string(index=False)
)

print("\nDONE!")
