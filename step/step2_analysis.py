import pandas as pd

INPUT_FILE = "results/amazonhelp_conversations.csv"

print("Loading AmazonHelp conversations...")

df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print("BASIC INFORMATION")
print("========================================")

print("Rows:", len(df))
print("Unique conversations:", df["conversation_id"].nunique())

print("\nSpeaker distribution:")
print(df["speaker"].value_counts())


# ------------------------------------------------------------
# Conversation length
# ------------------------------------------------------------

conversation_sizes = (
    df.groupby("conversation_id")
      .size()
)

print("\n========================================")
print("CONVERSATION LENGTH")
print("========================================")

print(conversation_sizes.describe())


# ------------------------------------------------------------
# Customer messages per conversation
# ------------------------------------------------------------

customer_counts = (
    df[df["speaker"] == "customer"]
    .groupby("conversation_id")
    .size()
)

print("\nCustomer messages per conversation:")
print(customer_counts.describe())


# ------------------------------------------------------------
# Support messages per conversation
# ------------------------------------------------------------

support_counts = (
    df[df["speaker"] == "support"]
    .groupby("conversation_id")
    .size()
)

print("\nSupport messages per conversation:")
print(support_counts.describe())


# ------------------------------------------------------------
# Find conversations containing BOTH customer and support
# ------------------------------------------------------------

speakers = (
    df.groupby("conversation_id")["speaker"]
      .unique()
)

valid_conversations = speakers[
    speakers.apply(
        lambda x: "customer" in x and "support" in x
    )
]

print("\n========================================")
print("VALID CONVERSATIONS")
print("========================================")

print(
    "Conversations containing both customer and support:",
    len(valid_conversations)
)


# ------------------------------------------------------------
# Show 5 example conversations
# ------------------------------------------------------------

print("\n========================================")
print("EXAMPLE CONVERSATIONS")
print("========================================")

example_ids = valid_conversations.index[:5]

for conversation_id in example_ids:

    print("\n----------------------------------------")
    print("Conversation:", conversation_id)
    print("----------------------------------------")

    conversation = df.loc[
        df["conversation_id"] == conversation_id
    ].sort_values(by=["created_at"])

    for _, row in conversation.iterrows():

        speaker = row["speaker"].upper()
        text = str(row["text"]).replace("\n", " ")

        print(f"{speaker}: {text}")


print("\nDONE!")