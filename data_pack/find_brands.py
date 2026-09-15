import pandas as pd

file_path = "./archive/twcs/twcs.csv"

print("Reading dataset...")

df = pd.read_csv(file_path, usecols=["author_id", "inbound"])

# Support accounts are identified by tweets that are not inbound.
support_tweets = df[df["inbound"] == False]

# Count tweets made by each support account.
brand_counts = support_tweets["author_id"].value_counts()

print("\n========== BRANDS / SUPPORT ACCOUNTS ==========")
print(brand_counts.to_string())

print("\n========== TOTAL SUPPORT ACCOUNTS ==========")
print(len(brand_counts))