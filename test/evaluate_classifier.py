import pandas as pd
from classifier import classify


INPUT_FILE = "results/amazonhelp_golden_labeled.csv"
OUTPUT_FILE = "results/classification_results_200.csv"


df = pd.read_csv(INPUT_FILE)

results = []

for i, row in df.iterrows():

    print("Processing...")

    predicted = classify(row["customer_message"])

    results.append({
        "example_id": row["example_id"],
        "customer_message": row["customer_message"],
        "actual_intent": row["final_intent"],
        "predicted_intent": predicted,
        "correct": predicted == row["final_intent"]
    })


results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

accuracy = results_df["correct"].mean()

print("\n========== RESULTS ==========")
print(f"Total examples: {len(results_df)}")
print(f"Correct: {results_df['correct'].sum()}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Saved to: {OUTPUT_FILE}")