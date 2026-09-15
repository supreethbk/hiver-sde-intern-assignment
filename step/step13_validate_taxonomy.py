import pandas as pd

FILE = "results/amazonhelp_golden_labeled.csv"

df = pd.read_csv(FILE)

counts = df["final_intent"].value_counts()

print("=" * 60)
print("TAXONOMY VALIDATION")
print("=" * 60)

for intent, count in counts.items():
    if count < 5:
        status = "⚠️ LOW"
    elif count < 10:
        status = "⚠️ SMALL"
    else:
        status = "✓ OK"

    print(f"{intent:<28} {count:>3}   {status}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("Total examples:", len(df))
print("Unique intents:", len(counts))
print("Intents with <5 examples:", (counts < 5).sum())
print("Intents with 5-9 examples:", ((counts >= 5) & (counts < 10)).sum())
print("Intents with >=10 examples:", (counts >= 10).sum())