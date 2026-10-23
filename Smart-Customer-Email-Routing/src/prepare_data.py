
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "tickets_english.csv"


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # Chi su dung email tieng Anh
    df = df[df["language"] == "en"].copy()

    # Chuyen gia tri thieu cua subject thanh chuoi rong
    df["subject"] = df["subject"].fillna("")

    # Ket hop tieu de va noi dung email
    df["text"] = (
        df["subject"].str.strip()
        + " "
        + df["body"].fillna("").str.strip()
    ).str.strip()

    # Chi giu cac cot can thiet cho bai toan dau tien
    df = df[["text", "queue"]].copy()

    # Loai bo cac dong khong co noi dung hoac nhan
    df = df[
        df["text"].ne("")
        & df["queue"].notna()
    ].copy()

    # Luu dataset da chuan bi
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=== KET QUA CHUAN BI DU LIEU ===")
    print("So email:", len(df))
    print("So bo phan:", df["queue"].nunique())

    print("\nPhan bo email theo bo phan:")
    print(df["queue"].value_counts().to_string())

    print("\nVi du du lieu:")
    print(df.head(3).to_string(index=False))

    print("\nDa luu tai:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
