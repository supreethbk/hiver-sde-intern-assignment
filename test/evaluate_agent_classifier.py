import pandas as pd
from agent_classifier import classify

GOLDEN_FILE = "results/amazonhelp_golden_labeled.csv"
OUTPUT_FILE = "results/agent_classification_results_200.csv"

df = pd.read_csv(GOLDEN_FILE)

results = []

for _, row in df.iterrows():

    message = row["customer_message"]
    actual = row["final_intent"]

    predicted = classify(message)

    results.append({
        "example_id": row["example_id"],
        "customer_message": message,
        "actual_intent": actual,
        "predicted_intent": predicted,
        "correct": predicted == actual
    })

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

correct_count = results_df["correct"].sum()
accuracy = results_df["correct"].mean()

print("===================================")
print("Improved Agent Classifier Results")
print("===================================")
print("Total:", len(results_df))
print("Correct:", correct_count)
print("Accuracy:", f"{accuracy:.2%}")
print("Results saved to:", OUTPUT_FILE)