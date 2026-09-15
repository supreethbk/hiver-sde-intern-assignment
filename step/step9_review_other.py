import pandas as pd

df = pd.read_csv("results/intent_manual_review.csv")

other = df[df["intent"] == "OTHER"]

print("=" * 60)
print("OTHER INTENT EXAMPLES")
print("=" * 60)
print("Total:", len(other))

for _, row in other.iterrows():
    print("\nExample:", row["example_id"])
    print("Customer:", row["customer_message"])
    print("Support :", row["support_response"])
    print("-" * 60)