import pandas as pd
import re
from collections import Counter

INPUT_FILE = "results/amazonhelp_pairs_analyzed.csv"

print("Loading AmazonHelp data...")

df = pd.read_csv(INPUT_FILE)

# ------------------------------------------------------------
# 1. Get unique customer messages
# ------------------------------------------------------------

customers = df[
    ["conversation_id", "customer_tweet_id", "customer_message"]
].drop_duplicates(
    subset=["customer_tweet_id"]
).copy()

customers["customer_message"] = (
    customers["customer_message"]
    .fillna("")
    .map(lambda x: str(x).strip())
)

print("\n========================================")
print("CUSTOMER DATA")
print("========================================")

print("Unique customer messages:", len(customers))


# ------------------------------------------------------------
# 2. Remove Twitter usernames and URLs for analysis
# ------------------------------------------------------------

def clean_text(text):
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("&amp;", "and")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


customers["clean_text"] = customers["customer_message"].map(clean_text)


# ------------------------------------------------------------
# 3. Count common words
# ------------------------------------------------------------

stopwords = {
    "the", "and", "for", "you", "that", "this",
    "with", "have", "was", "are", "but", "not",
    "my", "your", "from", "they", "what", "can",
    "how", "why", "just", "has", "had", "its",
    "our", "please", "amazon", "help", "about",
    "been", "would", "could", "will", "there",
    "when", "who", "all", "get", "got", "did",
    "didnt", "dont", "im", "ive", "its", "to",
    "of", "in", "on", "is", "a", "an", "i",
    "it", "me", "we", "us", "or", "do"
}

word_counter = Counter()

for text in customers["clean_text"]:

    words = text.split()

    for word in words:

        if len(word) >= 3 and word not in stopwords:
            word_counter[word] += 1


print("\n========================================")
print("MOST COMMON CUSTOMER WORDS")
print("========================================")

for word, count in word_counter.most_common(50):

    print(f"{word:25} {count}")


# ------------------------------------------------------------
# 4. Search for important support topics
# ------------------------------------------------------------

topics = {
    "delivery": [
        "delivery", "delivered", "deliver", "shipping",
        "shipment", "courier", "driver", "arrive", "late"
    ],

    "missing_package": [
        "missing", "not received", "didn't receive",
        "didnt receive", "never arrived", "lost package"
    ],

    "refund": [
        "refund", "money back", "refunded", "reimburse"
    ],

    "payment": [
        "payment", "charged", "charge", "debit",
        "credit card", "transaction", "billing"
    ],

    "return": [
        "return", "replacement", "replace", "returned"
    ],

    "account_login": [
        "login", "log in", "password", "account",
        "locked", "sign in", "email"
    ],

    "prime": [
        "prime", "membership", "subscription"
    ],

    "device": [
        "kindle", "echo", "alexa", "fire tv",
        "tablet", "device"
    ]
}


print("\n========================================")
print("TOPIC FREQUENCY")
print("========================================")

for topic, keywords in topics.items():

    count = 0

    for text in customers["clean_text"]:

        if any(keyword in text for keyword in keywords):
            count += 1

    percentage = (count / len(customers)) * 100

    print(
        f"{topic:20} "
        f"{count:6} messages "
        f"({percentage:.2f}%)"
    )


# ------------------------------------------------------------
# 5. Show examples for each topic
# ------------------------------------------------------------

print("\n========================================")
print("REPRESENTATIVE EXAMPLES")
print("========================================")

for topic, keywords in topics.items():

    print("\n")
    print("########################################")
    print("TOPIC:", topic.upper())
    print("########################################")

    found = 0

    for _, row in customers.iterrows():

        text = row["clean_text"]

        if any(keyword in text for keyword in keywords):

            print("\n-", row["customer_message"])

            found += 1

            if found >= 5:
                break


# ------------------------------------------------------------
# 6. Save exploration dataset
# ------------------------------------------------------------

OUTPUT_FILE = "results/amazonhelp_intent_exploration.csv"

customers.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("SAVED")
print("========================================")

print(OUTPUT_FILE)

print("\nDONE!")
