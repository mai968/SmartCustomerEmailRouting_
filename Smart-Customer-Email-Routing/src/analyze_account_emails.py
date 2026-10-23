
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
OUTPUT_PATH = BASE_DIR / "reports" / "account_related_emails.csv"

# Keywords related to login, accounts, and passwords
KEYWORDS = [
    "login",
    "log in",
    "sign in",
    "signin",
    "account",
    "password",
    "reset password",
    "forgot password",
    "unable to access",
    "cannot access",
    "can't access",
    "locked out",
    "authentication",
    "verification code",
]

def main():
    print("Loading training data...")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)

    # Validate required columns
    required_columns = {"text", "queue"}
    if not required_columns.issubset(df.columns):
        raise ValueError("train.csv must contain 'text' and 'queue' columns.")

    df["text"] = df["text"].fillna("").astype(str)
    df["queue"] = df["queue"].fillna("Unknown")

    # Find emails containing at least one keyword
    pattern = "|".join(
        pd.Series(KEYWORDS).str.replace(
            r"([.*+?^${}()|[\]\\])", r"\\\1", regex=True
        )
    )

    mask = df["text"].str.contains(pattern, case=False, na=False, regex=True)
    matched = df.loc[mask, ["text", "queue"]].copy()

    # Summarize queue distribution
    summary = (
        matched["queue"]
        .value_counts()
        .rename_axis("queue")
        .reset_index(name="email_count")
    )

    if len(matched) > 0:
        summary["percentage"] = (
            summary["email_count"] / len(matched) * 100
        ).round(2)

    # Save matched emails for manual inspection
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 55)
    print("ACCOUNT-RELATED EMAIL ANALYSIS")
    print("=" * 55)
    print(f"Total training emails: {len(df)}")
    print(f"Matched emails: {len(matched)}")

    print("\nQueue distribution among matched emails:")
    if matched.empty:
        print("No matching emails found. Consider expanding the keywords.")
    else:
        print(summary.to_string(index=False))

    print(f"\nMatched emails saved to:\n{OUTPUT_PATH}")

    print("\nSample matched emails:")
    for _, row in matched.head(10).iterrows():
        text = " ".join(row["text"].split())
        print(f"\nQueue: {row['queue']}")
        print(f"Text: {text[:250]}")

if __name__ == "__main__":
    main()
