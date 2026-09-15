import pandas as pd
import re

from reply_generator import safety_check


INPUT_FILE = "results/reply_generation_eval_20.csv"
OUTPUT_FILE = "results/reply_leakage_results_20.csv"


# ============================================================
# Load generated replies
# ============================================================

df = pd.read_csv(INPUT_FILE)

results = []

print("Running deterministic leakage evaluation...")
print(f"Examples: {len(df)}")


# ============================================================
# Check every reply
# ============================================================

for _, row in df.iterrows():

    reply = str(row["generated_reply"])

    violations = safety_check(reply)

    # --------------------------------------------------------
    # Additional checks
    # --------------------------------------------------------

    # Historical Twitter URL
    has_twitter_url = bool(
        re.search(
            r"https?://t\.co/\S+",
            reply,
            re.IGNORECASE
        )
    )

    # Any URL
    has_url = bool(
        re.search(
            r"https?://|www\.",
            reply,
            re.IGNORECASE
        )
    )

    # Placeholder
    has_placeholder = bool(
        re.search(
            r"\[[^\]]+\]",
            reply
        )
    )

    # Twitter username
    has_username = bool(
        re.search(
            r"@[A-Za-z0-9_]+",
            reply
        )
    )

    # Agent initials
    has_agent_initials = bool(
        re.search(
            r"\^[A-Z]{1,4}\b",
            reply
        )
    )

    # Signature leakage
    signature_terms = [
        "Amazon Customer Support",
        "Your Name",
        "Your Job Title",
        "Best regards"
    ]

    has_signature = any(
        term.lower() in reply.lower()
        for term in signature_terms
    )

    leaked = (
        has_url
        or has_placeholder
        or has_username
        or has_agent_initials
        or has_signature
        or len(violations) > 0
    )

    results.append({
        "example_id": row["example_id"],
        "generated_reply": reply,
        "url": has_url,
        "twitter_url": has_twitter_url,
        "placeholder": has_placeholder,
        "username": has_username,
        "agent_initials": has_agent_initials,
        "signature": has_signature,
        "violations": ", ".join(violations),
        "leakage_detected": leaked
    })


# ============================================================
# Save
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Summary
# ============================================================

total = len(results_df)

leaked_count = results_df[
    results_df["leakage_detected"]
].shape[0]

clean_count = total - leaked_count


print("\n========================================")
print("Deterministic Leakage Summary")
print("========================================")

print(f"Total replies: {total}")
print(f"Clean replies: {clean_count}")
print(f"Replies with leakage: {leaked_count}")

if total > 0:

    leakage_rate = (
        leaked_count / total
    ) * 100

    clean_rate = (
        clean_count / total
    ) * 100

    print(
        f"Leakage rate: {leakage_rate:.1f}%"
    )

    print(
        f"Leakage-free rate: {clean_rate:.1f}%"
    )


print("\nViolation counts:")

for column, label in [
    ("url", "URLs"),
    ("twitter_url", "Twitter URLs"),
    ("placeholder", "Placeholders"),
    ("username", "Usernames"),
    ("agent_initials", "Agent initials"),
    ("signature", "Signatures")
]:

    count = results_df[column].sum()

    print(
        f"{label}: {count}"
    )


print(
    f"\nSaved to: {OUTPUT_FILE}"
)