
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    print("\n=== 1. KICH THUOC DATASET ===")
    print("So dong:", len(df))
    print("So cot:", len(df.columns))

    print("\n=== 2. CAC COT ===")
    print(df.columns.tolist())

    print("\n=== 3. GIA TRI THIEU ===")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2)
    })
    print(missing.to_string())

    print("\n=== 4. PHAN BO BO PHAN ===")
    print(df["queue"].value_counts().to_string())

    print("\n=== 5. PHAN BO MUC DO UU TIEN ===")
    print(df["priority"].value_counts().to_string())

    print("\n=== 6. PHAN BO NGON NGU ===")
    print(df["language"].value_counts().to_string())

    print("\n=== 7. KIEM TRA EMAIL TRUNG LAP ===")
    print("So dong trung lap toan bo:",
          df.duplicated().sum())

    print("So body trung lap:",
          df["body"].duplicated().sum())

    print("\n=== 8. KIEM TRA NHAN QUEUE ===")
    print("So queue bi thieu:", df["queue"].isna().sum())

    print("\n=== 9. KIEM TRA DO DAI EMAIL ===")
    print(df["body"].str.len().describe())

if __name__ == "__main__":
    main()
