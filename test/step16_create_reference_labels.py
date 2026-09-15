import pandas as pd

INPUT_FILE = "results/reference_labeling_pilot_100.csv"
OUTPUT_FILE = "results/reference_labeled_100.csv"

# Labels for examples 1–100
labels = [
    "SELLER",
    "CUSTOMER_SERVICE",
    "CUSTOMER_SERVICE",
    "RETURN",
    "FEEDBACK",
    "DELIVERY_DELAY",
    "DELIVERY_PROBLEM",
    "WEBSITE_OR_APP",
    "DELIVERY_DELAY",
    "DELIVERY_PROBLEM",

    "CUSTOMER_SERVICE",
    "PACKAGE_NOT_RECEIVED",
    "RESOLUTION_CONFIRMATION",
    "INSUFFICIENT_CONTEXT",
    "SELLER",
    "DELIVERY_DELAY",
    "REFUND",
    "DELIVERY_DELAY",
    "SELLER",
    "ORDER_CANCELLATION",

    "CUSTOMER_SERVICE",
    "INSUFFICIENT_CONTEXT",
    "DELIVERY_TRACKING",
    "DELIVERY_DELAY",
    "REFUND",
    "CASUAL_ENGAGEMENT",
    "DELIVERY_DELAY",
    "PRODUCT_INFORMATION",
    "DELIVERY_PROBLEM",
    "DELIVERY_PROBLEM",

    "DEVICE",
    "RESOLUTION_CONFIRMATION",
    "OTHER",
    "REFUND",
    "WEBSITE_OR_APP",
    "FEEDBACK",
    "FEEDBACK",
    "RESOLUTION_CONFIRMATION",
    "DELIVERY_TRACKING",
    "CASUAL_ENGAGEMENT",
    "INSUFFICIENT_CONTEXT",
    "SELLER",
    "FEEDBACK",
    "CASUAL_ENGAGEMENT",
    "ACCOUNT",
    "WEBSITE_OR_APP",
    "DELIVERY_DELAY",
    "CASUAL_ENGAGEMENT",
    "CASUAL_ENGAGEMENT",
    "FEEDBACK",
    "DELIVERY_DELAY",
    "REFUND",
    "PAYMENT",
    "DELIVERY_DELAY",
    "DELIVERY_TRACKING",
    "INSUFFICIENT_CONTEXT",
    "PAYMENT",
    "INSUFFICIENT_CONTEXT",
    "RESOLUTION_CONFIRMATION",
    "DEVICE",

    "DELIVERY_DELAY",
    "CUSTOMER_SERVICE",
    "RESOLUTION_CONFIRMATION",
    "RESOLUTION_CONFIRMATION",
    "INSUFFICIENT_CONTEXT",
    "PACKAGE_NOT_RECEIVED",
    "PACKAGE_NOT_RECEIVED",
    "CASUAL_ENGAGEMENT",
    "ORDER_DISPATCH",
    "DELIVERY_PROBLEM",
    "CUSTOMER_SERVICE",
    "PAYMENT",
    "OTHER",
    "DELIVERY_TRACKING",
    "CASUAL_ENGAGEMENT",
    "PAYMENT",
    "CUSTOMER_SERVICE",
    "DELIVERY_TRACKING",
    "CASUAL_ENGAGEMENT",
    "CUSTOMER_SERVICE",
    "DELIVERY_DELAY",
    "CUSTOMER_SERVICE",
    "OTHER",
    "PAYMENT",
    "ACCOUNT",
    "DELIVERY_DELAY",
    "ORDER_CANCELLATION",
    "OTHER",
    "CASUAL_ENGAGEMENT",
    "CUSTOMER_SERVICE",
    "DELIVERY_TRACKING",
    "WEBSITE_OR_APP",
    "CUSTOMER_SERVICE",
    "FEEDBACK",
    "DELIVERY_TRACKING",
    "SELLER",
    "FEEDBACK",
    "INSUFFICIENT_CONTEXT",
    "RESOLUTION_CONFIRMATION",
    "CASUAL_ENGAGEMENT",
]

# Read original pilot
df = pd.read_csv(INPUT_FILE)

if len(df) != 100:
    raise ValueError(f"Expected 100 rows, found {len(df)}")

if len(labels) != 100:
    raise ValueError(f"Expected 100 labels, found {len(labels)}")

# Add labels without modifying original file
df["intent"] = labels

# Save new labeled reference file
df.to_csv(OUTPUT_FILE, index=False)

print(f"Created: {OUTPUT_FILE}")
print(f"Rows: {len(df)}")
print(f"Unique intents: {df['intent'].nunique()}")
print("\nIntent counts:")
print(df["intent"].value_counts())