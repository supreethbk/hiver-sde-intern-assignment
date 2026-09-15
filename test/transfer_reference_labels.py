import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


REFERENCE_FILE = "results/amazonhelp_reference_3k.csv"
LABELED_FILE = "results/reference_labeled_100.csv"
OUTPUT_FILE = "results/amazonhelp_reference_3k_labeled.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

reference = pd.read_csv(REFERENCE_FILE)
labeled = pd.read_csv(LABELED_FILE)

reference["customer_message"] = (
    reference["customer_message"]
    .fillna("")
)

labeled["customer_message"] = (
    labeled["customer_message"]
    .fillna("")
)


# --------------------------------------------------
# Prepare labeled examples
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1
)

labeled_vectors = vectorizer.fit_transform(
    labeled["customer_message"]
)

reference_vectors = vectorizer.transform(
    reference["customer_message"]
)


# --------------------------------------------------
# Find nearest labeled example
# --------------------------------------------------

similarities = (reference_vectors @ labeled_vectors.T).toarray()


best_indices = np.argmax(
    similarities,
    axis=1
)

best_scores = np.max(
    similarities,
    axis=1
)


# --------------------------------------------------
# Transfer intent
# --------------------------------------------------

reference["intent"] = [
    labeled.iloc[index]["intent"]
    for index in best_indices
]

reference["label_similarity"] = best_scores


# --------------------------------------------------
# Save
# --------------------------------------------------

reference.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Done.")
print("Reference examples:", len(reference))
print("Saved:", OUTPUT_FILE)

print("\nIntent distribution:")
print(
    reference["intent"]
    .value_counts()
)

print("\nSimilarity statistics:")
print(
    reference["label_similarity"].describe()
)