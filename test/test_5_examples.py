import pandas as pd
from classifier import classify


df = pd.read_csv("results/amazonhelp_golden_labeled.csv")

test_data = df.head(5)

for _, row in test_data.iterrows():

    prediction = classify(row["customer_message"])

    print("\nExample:", row["example_id"])
    print("Customer:", row["customer_message"])
    print("Actual:", row["final_intent"])
    print("Predicted:", prediction)