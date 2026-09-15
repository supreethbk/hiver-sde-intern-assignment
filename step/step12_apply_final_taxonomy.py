import pandas as pd

INPUT_FILE = "results/intent_manual_review.csv"
OUTPUT_FILE = "results/amazonhelp_golden_labeled.csv"

df = pd.read_csv(INPUT_FILE)

mapping = {
    "DELIVERY_DELAY": "DELIVERY_DELAY",
    "PACKAGE_NOT_RECEIVED": "PACKAGE_NOT_RECEIVED",

    "DELIVERY_ISSUE": "DELIVERY_PROBLEM",
    "WRONG_DELIVERY": "DELIVERY_PROBLEM",
    "PRIME_DELIVERY_ISSUE": "DELIVERY_PROBLEM",

    "DELIVERY_INSTRUCTIONS": "DELIVERY_INSTRUCTIONS",
    "DELIVERY_ADDRESS_ISSUE": "DELIVERY_INSTRUCTIONS",
    "ADDRESS_ISSUE": "DELIVERY_INSTRUCTIONS",

    "TRACKING_ISSUE": "DELIVERY_TRACKING",
    "DELIVERY_ESTIMATE": "DELIVERY_TRACKING",

    "ORDER_NOT_DISPATCHED": "ORDER_DISPATCH",
    "ORDER_ON_HOLD": "ORDER_DISPATCH",
    "ORDER_NOT_FULFILLED": "ORDER_DISPATCH",

    "ORDER_CANCELLATION": "ORDER_CANCELLATION",
    "RETURN_ISSUE": "RETURN",
    "REFUND_ISSUE": "REFUND",

    "PAYMENT_ISSUE": "PAYMENT",
    "PAYMENT_PROBLEM": "PAYMENT",
    "PRIME_CHARGE_ISSUE": "PAYMENT",

    "ACCOUNT_PROBLEM": "ACCOUNT",
    "ACCOUNT_SECURITY": "ACCOUNT",
    "FRAUD_SECURITY_ISSUE": "ACCOUNT",

    "SELLER_ISSUE": "SELLER",
    "PRODUCT_INFORMATION": "PRODUCT_INFORMATION",
    "DIGITAL_DEVICE_ISSUE": "DEVICE",

    "LINK_OR_WEBSITE_ISSUE": "WEBSITE_OR_APP",

    "CUSTOMER_SERVICE": "CUSTOMER_SERVICE",
    "CALLBACK_ISSUE": "CUSTOMER_SERVICE",

    "FEEDBACK": "FEEDBACK",
    "REVIEW_ISSUE": "FEEDBACK",

    "OTHER": "OTHER",
    "CLAIM_ISSUE": "OTHER",
    "CODE_NOT_RECEIVED": "OTHER",
    "EMAIL_ISSUE": "OTHER",
    "PRICE_ISSUE": "OTHER",
    "PRICE_MATCH_ISSUE": "OTHER",
    "PACKAGING_ISSUE": "OTHER",
    "PRIME_MEMBERSHIP": "OTHER",
    "ORDER_PROBLEM": "OTHER",
    "CONTEST_ISSUE": "OTHER",
}

# These were originally labeled OTHER,
# but our manual review identified more useful categories.
other_special = {
    8: "DELIVERY_TRACKING",
    9: "DELIVERY_TRACKING",

    12: "RESOLUTION_CONFIRMATION",
    15: "RESOLUTION_CONFIRMATION",
    88: "RESOLUTION_CONFIRMATION",
    98: "RESOLUTION_CONFIRMATION",
    99: "RESOLUTION_CONFIRMATION",
    107: "RESOLUTION_CONFIRMATION",
    131: "RESOLUTION_CONFIRMATION",
    133: "RESOLUTION_CONFIRMATION",
    143: "RESOLUTION_CONFIRMATION",
    147: "RESOLUTION_CONFIRMATION",
    152: "RESOLUTION_CONFIRMATION",

    72: "INSUFFICIENT_CONTEXT",
    74: "INSUFFICIENT_CONTEXT",
    92: "INSUFFICIENT_CONTEXT",
    95: "INSUFFICIENT_CONTEXT",
    97: "INSUFFICIENT_CONTEXT",
    102: "INSUFFICIENT_CONTEXT",
    138: "INSUFFICIENT_CONTEXT",

    109: "CASUAL_ENGAGEMENT",
    115: "CASUAL_ENGAGEMENT",
    180: "CASUAL_ENGAGEMENT",
    185: "CASUAL_ENGAGEMENT",
    189: "CASUAL_ENGAGEMENT",

    181: "OTHER",
}

new_labels = []

for _, row in df.iterrows():

    example_id = int(row["example_id"])
    old_label = row["intent"]

    if old_label == "OTHER" and example_id in other_special:
        new_label = other_special[example_id]
    else:
        new_label = mapping.get(old_label, "OTHER")

    new_labels.append(new_label)

df["final_intent"] = new_labels

df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("FINAL TAXONOMY APPLIED")
print("=" * 60)

print("Examples:", len(df))
print("Original file preserved:", INPUT_FILE)
print("New file:", OUTPUT_FILE)

print("\nFINAL INTENT FREQUENCY")
print("=" * 60)
print(df["final_intent"].value_counts())

print("\nUNIQUE FINAL INTENTS:", df["final_intent"].nunique())