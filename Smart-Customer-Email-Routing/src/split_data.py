from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Xac dinh thu muc project
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Duong dan dataset da chuan bi
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "tickets_english.csv"
)

# Thu muc luu ket qua
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def main():
    # Doc dataset
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # Buoc 1: tach 80% train, 20% con lai (temp)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["queue"],  # giu ty le cac queue giong nhau giua cac tap
    )

    # Buoc 2: chia doi temp thanh 10% validation, 10% test (so voi tong)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["queue"],
    )

    # Luu ba tap du lieu
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

    # In ket qua
    print("=== KET QUA CHIA DU LIEU ===")
    print("Tong so email:", len(df))
    print(f"Training set:   {len(train_df)} ({len(train_df) / len(df):.1%})")
    print(f"Validation set: {len(val_df)} ({len(val_df) / len(df):.1%})")
    print(f"Test set:       {len(test_df)} ({len(test_df) / len(df):.1%})")

    print("\nPhan bo queue trong training set:")
    print(train_df["queue"].value_counts().to_string())

    print("\nPhan bo queue trong validation set:")
    print(val_df["queue"].value_counts().to_string())

    print("\nPhan bo queue trong test set:")
    print(test_df["queue"].value_counts().to_string())

    print("\nDa luu train.csv, val.csv va test.csv tai:", OUTPUT_DIR)


if __name__ == "__main__":
    main()