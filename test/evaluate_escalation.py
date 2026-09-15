import pandas as pd
from escalation_decision import decide_escalation


INPUT_FILE = "results/reply_generation_eval_20.csv"
OUTPUT_FILE = "results/escalation_results_reply_20.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    results = []

    for _, row in df.iterrows():
        decision = decide_escalation(
            row["golden_intent"],
            row["generated_reply"]
        )

        results.append({
            "example_id": row["example_id"],
            "customer_message": row["customer_message"],
            "intent": row["golden_intent"],
            "generated_reply": row["generated_reply"],
            "decision": decision["decision"],
            "reason": decision["reason"],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 50)
    print("Escalation Evaluation with Generated Replies")
    print("=" * 50)

    print(f"\nTotal examples: {len(results_df)}")

    print("\nDecision distribution:")
    print(results_df["decision"].value_counts())

    print("\nEscalated examples:")
    escalated = results_df[results_df["decision"] == "ESCALATE"]

    for _, row in escalated.iterrows():
        print(f"\nExample {row['example_id']}")
        print(f"Intent: {row['intent']}")
        print(f"Reply: {row['generated_reply']}")
        print(f"Reason: {row['reason']}")

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()