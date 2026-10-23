
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"

INPUT_PATH = REPORT_DIR / "svm_test_predictions.csv"
OUTPUT_PATH = REPORT_DIR / "svm_test_errors.csv"


def main():
    # Load predictions from the test set
    df = pd.read_csv(INPUT_PATH)

    # Keep only incorrectly classified emails
    errors = df[df["correct"] == False].copy()

    # Count each actual queue -> predicted queue combination
    error_pairs = (
        errors.groupby(["actual_queue", "predicted_queue"])
        .size()
        .reset_index(name="error_count")
        .sort_values("error_count", ascending=False)
    )

    # Save error details
    errors.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)
    print("MODEL ERROR ANALYSIS")
    print("=" * 60)
    print(f"Total test emails: {len(df)}")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"Error rate: {len(errors) / len(df):.2%}")

    print("\nTOP 10 CONFUSION PAIRS")
    print(error_pairs.head(10).to_string(index=False))

    print(f"\nError details saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
