import pandas as pd

from reply_generator import retrieve
from reply_generator import generate_safe_reply

INPUT_FILE = "results/amazonhelp_golden_labeled.csv"
OUTPUT_FILE = "results/reply_generation_eval_20.csv"


# ------------------------------------------------------------
# Load Golden Set
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

# Use a fixed sample so results are reproducible
sample = df.sample(
    n=20,
    random_state=42
).copy()


results = []


# ------------------------------------------------------------
# Generate replies
# ------------------------------------------------------------

for i, row in sample.iterrows():

    customer_message = row["customer_message"]

    print("\n========================================")
    print(f"Example {row['example_id']}")
    print("========================================")

    print("Customer:")
    print(customer_message)

    try:

        # Retrieve historical cases
        cases = retrieve(
            customer_message,
            top_k=5
        )

        # Generate reply
        reply, violations = generate_safe_reply(
            customer_message,
            cases   
        )

        print("\nGenerated reply:")
        print(reply)

        results.append({
            "example_id": row["example_id"],
            "customer_message": customer_message,
            "golden_intent": row["final_intent"],
            "generated_reply": reply
        })

    except Exception as e:

        print("\nERROR:")
        print(e)

        results.append({
            "example_id": row["example_id"],
            "customer_message": customer_message,
            "golden_intent": row["final_intent"],
            "generated_reply": "",
            "error": str(e)
        })


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("Evaluation complete")
print("========================================")

print(f"Examples evaluated: {len(results_df)}")
print(f"Saved to: {OUTPUT_FILE}")