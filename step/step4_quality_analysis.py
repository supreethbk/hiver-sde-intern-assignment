import pandas as pd

INPUT_FILE = "results/amazonhelp_customer_support_pairs.csv"

print("Loading customer-support pairs...")

df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print("BASIC INFORMATION")
print("========================================")

print("Total pairs:", len(df))
print("Unique conversations:", df["conversation_id"].nunique())
print("Unique customer messages:", df["customer_tweet_id"].nunique())


# ------------------------------------------------------------
# Message length analysis
# ------------------------------------------------------------

df["customer_message"] = df["customer_message"].fillna("")
df["support_response"] = df["support_response"].fillna("")

df["customer_length"] = df["customer_message"].map(
    lambda x: len(str(x))
)

df["support_length"] = df["support_response"].map(
    lambda x: len(str(x))
)


print("\n========================================")
print("MESSAGE LENGTH")
print("========================================")

print("\nCustomer message length:")
print(df["customer_length"].describe())

print("\nSupport response length:")
print(df["support_length"].describe())


# ------------------------------------------------------------
# Very short customer messages
# ------------------------------------------------------------

print("\n========================================")
print("VERY SHORT CUSTOMER MESSAGES")
print("========================================")

short_customer = df[
    df["customer_length"] <= 15
]

print(
    "Customer messages <= 15 characters:",
    len(short_customer)
)

print("\nExamples:")

for text in short_customer["customer_message"].head(20):
    print("-", text)


# ------------------------------------------------------------
# Very short support responses
# ------------------------------------------------------------

print("\n========================================")
print("VERY SHORT SUPPORT RESPONSES")
print("========================================")

short_support = df[
    df["support_length"] <= 30
]

print(
    "Support responses <= 30 characters:",
    len(short_support)
)

print("\nExamples:")

for text in short_support["support_response"].head(20):
    print("-", text)


# ------------------------------------------------------------
# Common generic support responses
# ------------------------------------------------------------

print("\n========================================")
print("COMMON SUPPORT RESPONSES")
print("========================================")

common_responses = (
    df["support_response"]
    .value_counts()
    .head(20)
)

print(common_responses.to_string())


# ------------------------------------------------------------
# Duplicate customer messages
# ------------------------------------------------------------

print("\n========================================")
print("DUPLICATE CUSTOMER MESSAGES")
print("========================================")

duplicate_counts = (
    df["customer_message"]
    .value_counts()
)

duplicates = duplicate_counts[
    duplicate_counts > 1
]

print(
    "Customer messages appearing more than once:",
    len(duplicates)
)

print("\nExamples:")

print(
    duplicates.head(20).to_string()
)


# ------------------------------------------------------------
# Duplicate support responses
# ------------------------------------------------------------

print("\n========================================")
print("DUPLICATE SUPPORT RESPONSES")
print("========================================")

support_duplicates = (
    df["support_response"]
    .value_counts()
)

support_duplicates = support_duplicates[
    support_duplicates > 1
]

print(
    "Support responses appearing more than once:",
    len(support_duplicates)
)

print("\nExamples:")

print(
    support_duplicates.head(20).to_string()
)


# ------------------------------------------------------------
# Save analysis-ready copy
# ------------------------------------------------------------

OUTPUT_FILE = "results/amazonhelp_pairs_analyzed.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("SAVED")
print("========================================")

print(OUTPUT_FILE)

print("\nDONE!")
