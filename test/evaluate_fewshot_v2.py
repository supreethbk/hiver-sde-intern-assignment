import pandas as pd
from fewshot_classifier_v2 import classify
from openai import OpenAI

INPUT_FILE = "results/amazonhelp_golden_labeled.csv"
OUTPUT_FILE = "results/fewshot_v2_classification_results_200.csv"

df = pd.read_csv(INPUT_FILE)

results = []

for _, row in df.iterrows():
    predicted = classify(row["customer_message"])
    actual = row["final_intent"]

    results.append({
        "example_id": row["example_id"],
        "customer_message": row["customer_message"],
        "actual_intent": actual,
        "predicted_intent": predicted,
        "correct": actual == predicted
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

correct = results_df["correct"].sum()
total = len(results_df)
accuracy = correct / total * 100

print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Saved: {OUTPUT_FILE}")