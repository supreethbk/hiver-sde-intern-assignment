import pandas as pd

df = pd.read_csv("results/intent_manual_review.csv")

print("\nINTENT FREQUENCY")
print("=" * 40)

counts = df["intent"].value_counts()

print(counts)
print("\nTotal labeled:", counts.sum())