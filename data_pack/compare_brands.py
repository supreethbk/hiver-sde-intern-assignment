import pandas as pd
import os

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

FILE_PATH = "data/raw/twcs.csv"

# Top brands from our previous analysis
BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways"
]

# Number of support tweets to collect per brand
SAMPLE_PER_BRAND = 100

OUTPUT_DIR = "data/brand_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# READ DATA IN CHUNKS
# --------------------------------------------------

print("Reading dataset in chunks...")

brand_data = {brand: [] for brand in BRANDS}

for chunk in pd.read_csv(
    FILE_PATH,
    chunksize=100000,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
):

    # Find tweets belonging to our candidate brands
    matches = chunk[
        chunk["author_id"].isin(BRANDS)
    ]

    for brand in BRANDS:
        rows = matches[matches["author_id"] == brand]

        if len(brand_data[brand]) < SAMPLE_PER_BRAND:
            remaining = SAMPLE_PER_BRAND - len(brand_data[brand])
            brand_data[brand].extend(
                rows.head(remaining).to_dict("records")
            )

    # Stop once we have enough from every brand
    if all(
        len(brand_data[brand]) >= SAMPLE_PER_BRAND
        for brand in BRANDS
    ):
        break

# --------------------------------------------------
# SAVE SAMPLES
# --------------------------------------------------

print("\n========== SAMPLE RESULTS ==========")

for brand in BRANDS:

    df = pd.DataFrame(brand_data[brand])

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{brand}_sample.csv"
    )

    df.to_csv(output_file, index=False)

    print(f"{brand}: {len(df)} tweets -> {output_file}")

print("\nDone!")
print(f"\nFiles saved inside: {OUTPUT_DIR}/")
