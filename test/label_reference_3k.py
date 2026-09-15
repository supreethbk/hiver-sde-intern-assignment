import pandas as pd
import time

from classifier import classify


INPUT_FILE = "results/amazonhelp_reference_3k.csv"
OUTPUT_FILE = "results/amazonhelp_reference_3k_labeled.csv"


df = pd.read_csv(INPUT_FILE)

df["customer_message"] = df["customer_message"].fillna("")

predicted_intents = []

total = len(df)

print(f"Total messages: {total}")

start_time = time.time()

for i, message in enumerate(df["customer_message"], 1):

    try:
        intent = classify(message)
        predicted_intents.append(intent)

    except Exception as e:
        print(f"\nError at row {i}: {e}")
        predicted_intents.append("OTHER")

    if i % 50 == 0:

        elapsed = time.time() - start_time

        print(
            f"Processed {i}/{total} "
            f"({i / total * 100:.1f}%) "
            f"Elapsed: {elapsed / 60:.1f} min"
        )


df["intent"] = predicted_intents

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nDone.")
print("Saved:", OUTPUT_FILE)
print("\nIntent distribution:")
print(df["intent"].value_counts())