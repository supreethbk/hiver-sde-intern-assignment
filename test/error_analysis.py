import pandas as pd

x = pd.read_csv("results/classification_results_200.csv")

errors = x[x["correct"] == False]

print("\nTOP CONFUSIONS:\n")

print(
    errors
    .groupby(["actual_intent", "predicted_intent"])
    .size()
    .sort_values(ascending=False)
    .head(20)
    .to_string()
)
