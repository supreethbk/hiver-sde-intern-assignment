import pandas as pd
from collections import defaultdict
import re

FILE_PATH = "data/raw/twcs.csv"

# Start with the same dataset slice used in our previous analysis
NROWS = 500000

SAMPLE_PER_INTENT = 30

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

# ---------------------------------------------------------
# Convert IDs to numbers
# ---------------------------------------------------------

df["tweet_id"] = pd.to_numeric(
    df["tweet_id"],
    errors="coerce"
)

df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"],
    errors="coerce"
)

# Remove invalid tweet IDs
df = df.dropna(
    subset=["tweet_id"]
)

print("Tweets loaded:", len(df))

# ---------------------------------------------------------
# Create lookup tables
# ---------------------------------------------------------

tweets = df.set_index(
    "tweet_id"
).to_dict("index")

children = defaultdict(list)

for _, row in df.iterrows():

    tweet_id = row["tweet_id"]

    parent_id = row[
        "in_response_to_tweet_id"
    ]

    if pd.notna(parent_id):

        children[parent_id].append(
            tweet_id
        )

print(
    "Parent-child relationships:",
    sum(len(v) for v in children.values())
)

# ---------------------------------------------------------
# Find conversation roots
# ---------------------------------------------------------

roots = []

for _, row in df.iterrows():

    tweet_id = row["tweet_id"]

    parent_id = row[
        "in_response_to_tweet_id"
    ]

    if (
        pd.isna(parent_id)
        or parent_id not in tweets
    ):

        roots.append(tweet_id)

print(
    "Conversation roots found:",
    len(roots)
)

# ---------------------------------------------------------
# Candidate final taxonomy
# ---------------------------------------------------------

intent_keywords = {

    "delivery_issue": [
        "late",
        "delayed",
        "delay",
        "delivery",
        "not arrived",
        "not delivered",
        "still waiting",
        "out for delivery",
        "scheduled delivery"
    ],

    "package_missing": [
        "delivered but",
        "marked delivered",
        "says delivered",
        "didn't receive",
        "did not receive",
        "not received",
        "missing package",
        "lost package",
        "lost parcel"
    ],

    "return_replacement": [
        "return",
        "returned",
        "returning",
        "return pickup",
        "pickup",
        "replace",
        "replacement",
        "exchange"
    ],

    "refund_issue": [
        "refund",
        "refunded",
        "money back",
        "reimbursement",
        "refund pending",
        "refund status"
    ],

    "payment_charge_issue": [
        "payment",
        "charged",
        "charge",
        "payment failed",
        "charged twice",
        "charged three times",
        "credit card",
        "debit card"
    ],

    "prime_membership": [
        "prime",
        "prime membership",
        "prime member",
        "subscription",
        "membership",
        "cancel prime"
    ],

    "account_login": [
        "account",
        "login",
        "log in",
        "locked",
        "password",
        "sign in",
        "verification code",
        "2step",
        "two step"
    ],

    "digital_device": [
        "prime video",
        "video",
        "echo",
        "alexa",
        "kindle",
        "fire tv",
        "firestick",
        "fire tablet",
        "streaming"
    ]
}

# ---------------------------------------------------------
# Reconstruct conversations
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

        visited.add(
            current_id
        )

        conversation_ids.append(
            current_id
        )

        for child_id in children.get(
            current_id,
            []
        ):

            queue.append(
                child_id
            )

    # Ignore very small conversations
    if len(conversation_ids) < 2:
        continue

    # Ignore extremely large threads
    if len(conversation_ids) > 20:
        continue

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

    messages.sort(
        key=lambda x: x["created_at"]
    )

    conversations.append({
        "root_tweet_id": int(root_id),
        "message_count": len(messages),
        "messages": messages
    })

print(
    "Conversations reconstructed:",
    len(conversations)
)

# ---------------------------------------------------------
# Identify AmazonHelp conversations
#
# A conversation is considered AmazonHelp-related when
# it contains a support tweet whose author_id is AmazonHelp.
#
# The TWCS dataset's author_id field contains account IDs,
# so we first find AmazonHelp's author ID from the data
# using the support-account identification already used
# in our previous analysis.
# ---------------------------------------------------------

# Find AmazonHelp support tweets by checking the
# conversations' support authors against known brand samples.

# Load AmazonHelp tweets from the reconstructed conversations
# using the customer/support structure and the previously
# identified AmazonHelp author ID if available.

amazonhelp_author_ids = set()

# Look for support accounts with "AmazonHelp" in the author_id.
# If author_id itself is numeric, this won't work, so we use
# the AmazonHelp sample file to identify its support tweets.

try:

    amazon_sample = pd.read_csv(
        "data/brand_samples/AmazonHelp_sample.csv"
    )

    if "author_id" in amazon_sample.columns:

        amazonhelp_author_ids.update(
            amazon_sample.loc[
                amazon_sample["inbound"] == False,
                "author_id"
            ].dropna().unique()
        )

except FileNotFoundError:

    print(
        "AmazonHelp sample file not found."
    )

# ---------------------------------------------------------
# If the sample file is unavailable, identify AmazonHelp
# through the support author IDs from the dataset.
# ---------------------------------------------------------

if not amazonhelp_author_ids:

    print(
        "WARNING: AmazonHelp author ID could not "
        "be identified from the sample."
    )

    print(
        "Please make sure "
        "data/brand_samples/AmazonHelp_sample.csv exists."
    )

# ---------------------------------------------------------
# Extract AmazonHelp customer messages
# ---------------------------------------------------------

customer_messages = []

amazon_conversation_count = 0

for conversation in conversations:

    messages = conversation["messages"]

    contains_amazonhelp = False

    for message in messages:

        if (
            message["speaker"] == "SUPPORT"
            and message["author_id"]
            in amazonhelp_author_ids
        ):

            contains_amazonhelp = True

            break

    if not contains_amazonhelp:
        continue

    amazon_conversation_count += 1

    for message in messages:

        if message["speaker"] == "CUSTOMER":

            text = str(
                message["text"]
            ).strip()

            if not text:
                continue

            customer_messages.append({
                "tweet_id": message["tweet_id"],
                "text": text
            })

customers = pd.DataFrame(
    customer_messages,
    columns=[
        "tweet_id",
        "text"
    ]
)

customers = customers.drop_duplicates(
    subset=["text"]
)

print(
    "AmazonHelp conversations:",
    amazon_conversation_count
)

print(
    "Unique AmazonHelp customer messages:",
    len(customers)
)

# ---------------------------------------------------------
# Match candidate intents
# ---------------------------------------------------------

records = []

for intent, keywords in intent_keywords.items():

    pattern = "|".join(
        re.escape(keyword)
        for keyword in keywords
    )

    matches = customers[
        customers["text"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    ]

    matches = matches.drop_duplicates(
        subset=["text"]
    )

    matches = matches.head(
        SAMPLE_PER_INTENT
    )

    for _, row in matches.iterrows():

        records.append({
            "intent": intent,
            "tweet_id": row["tweet_id"],
            "text": row["text"]
        })

# ---------------------------------------------------------
# Save evidence
# ---------------------------------------------------------

evidence = pd.DataFrame(
    records,
    columns=[
        "intent",
        "tweet_id",
        "text"
    ]
)

output_file = (
    "results/"
    "amazon_taxonomy_verification.csv"
)

evidence.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)

# ---------------------------------------------------------
# Display examples
# ---------------------------------------------------------

for intent in intent_keywords:

    print(
        "\n" + "=" * 80
    )

    print(
        intent.upper()
    )

    print(
        "=" * 80
    )

    subset = evidence[
        evidence["intent"] == intent
    ]

    if len(subset) == 0:

        print(
            "No examples found."
        )

        continue

    for i, (_, row) in enumerate(
        subset.iterrows(),
        start=1
    ):

        print(
            f"{i}. {row['text']}"
        )

print("\nDONE")
