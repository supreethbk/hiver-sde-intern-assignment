import pandas as pd

REFERENCE_FILE = "results/reference_labeled_100.csv"

df = pd.read_csv(REFERENCE_FILE)

print("Reference examples:", len(df))
print("\nIntent coverage:")

counts = df["intent"].value_counts().sort_values(ascending=False)

print(counts.to_string())

print("\nNumber of intents:", len(counts))