import pandas as pd
import re

FILE_PATH = "data/raw/twcs.csv"
NROWS = 500000
SAMPLE_PER_INTENT = 30

print("Loading dataset...")

df = pd.read_csv(
    FILE_PATH,
    nrows=NROWS,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text"
    ]
)

# Candidate intent groups from our previous analysis
intent_keywords = {
    "delivery_delay": [
        "late", "delayed", "delay",
        "not arrived", "not delivered",
        "still waiting", "delivery"
    ],

    "tracking_problem": [
        "tracking", "track",
        "tracking number",
        "tracking hasn't",
        "tracking hasnt",
        "not updating"
    ],

    "refund_problem": [
        "refund", "refunded",
        "money back", "credit",
        "reimbursement"
    ],

    "return_problem": [
        "return", "returned",
        "returning", "return pickup",
        "pickup"
    ],

    "wrong_or_damaged_product": [
        "wrong item", "wrong product",
        "wrong quantity", "damaged",
        "broken", "defective",
        "missing item"
    ],

    "payment_problem": [
        "payment", "charged", "charge",
        "credit card", "debit card",
        "payment failed"
    ],

    "prime_subscription": [
        "prime", "subscription",
        "membership", "cancel prime"
    ],

    "account_problem": [
        "account", "login",
        "password", "locked",
        "sign in"
    ],

    "amazon_video": [
        "prime video", "streaming",
        "movie", "episode"
    ],

    "amazon_device": [
        "echo", "alexa",
        "fire tv", "kindle",
        "fire tablet"
    ]
}

# ---------------------------------------------------------
# Customer messages that belong to Amazon conversations
# ---------------------------------------------------------

# We use messages that directly mention Amazon/AmazonHelp
# as a practical evidence sample for taxonomy discovery.

customers = df[
    (df["inbound"] == True) &
    (df["text"].notna())
].copy()

customers["text"] = customers["text"].astype(str).str.strip()

customers = customers[
    customers["text"].str.contains(
        r"AmazonHelp|Amazon",
        case=False,
        regex=True,
        na=False
    )
]

print(f"Amazon-related customer messages: {len(customers)}")

# ---------------------------------------------------------
# Collect examples
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

    # Remove duplicate messages
    matches = matches.drop_duplicates(
        subset=["text"]
    )

    # Take first N examples
    matches = matches.head(SAMPLE_PER_INTENT)

    for _, row in matches.iterrows():

        records.append({
            "intent": intent,
            "tweet_id": row["tweet_id"],
            "text": row["text"]
        })

evidence = pd.DataFrame(records)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output = "results/amazon_intent_evidence.csv"

evidence.to_csv(
    output,
    index=False
)

print("\nSaved:")
print(output)

# ---------------------------------------------------------
# Display examples
# ---------------------------------------------------------

for intent in intent_keywords:

    print("\n" + "=" * 80)
    print(intent.upper())
    print("=" * 80)

    subset = evidence[
        evidence["intent"] == intent
    ]

    for i, (_, row) in enumerate(
        subset.iterrows(),
        start=1
    ):
        print(f"{i}. {row['text']}")

print("\nDONE")