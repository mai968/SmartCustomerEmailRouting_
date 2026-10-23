
from pathlib import Path
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score


# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 2. Load train and validation data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    X_train_text = train_df["text"].fillna("").astype(str)
    y_train = train_df["queue"]

    X_val_text = val_df["text"].fillna("").astype(str)
    y_val = val_df["queue"]

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # 3. Convert text into TF-IDF vectors
    print("\nFitting TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    print(f"Number of TF-IDF features: {X_train.shape[1]}")

    # 4. Define SVM configurations
    # c_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    c_values = [5.0, 10.0, 20.0, 50.0]
    class_weight_values = [None, "balanced"]

    results = []

    # 5. Train and evaluate each configuration
    for class_weight in class_weight_values:
        for c_value in c_values:
            print(
                f"\nTraining Linear SVM: "
                f"C={c_value}, class_weight={class_weight}"
            )

            start_time = time.time()

            model = LinearSVC(
                C=c_value,
                class_weight=class_weight,
                random_state=42,
                max_iter=5000
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            accuracy = accuracy_score(y_val, y_pred)
            macro_f1 = f1_score(
                y_val, y_pred, average="macro", zero_division=0
            )
            weighted_f1 = f1_score(
                y_val, y_pred, average="weighted", zero_division=0
            )

            elapsed = time.time() - start_time

            results.append({
                "model": "Linear SVM",
                "C": c_value,
                "class_weight": str(class_weight),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "training_seconds": round(elapsed, 2)
            })

            print(
                f"Accuracy={accuracy:.4f} | "
                f"Macro F1={macro_f1:.4f} | "
                f"Weighted F1={weighted_f1:.4f} | "
                f"Time={elapsed:.2f}s"
            )

    # 6. Sort results by Macro F1
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="macro_f1", ascending=False
    ).reset_index(drop=True)

    # output_path = REPORT_DIR / "svm_tuning_results.csv"
    # results_df.to_csv(output_path, index=False)

    # 7. Display results
    print("\n" + "=" * 70)
    print("SVM TUNING RESULTS — RANKED BY MACRO F1")
    print("=" * 70)

    print(results_df.to_string(index=False, float_format="%.4f"))

    # print(f"\nResults saved to: {output_path}")

    best = results_df.iloc[0]
    print("\nBest configuration on validation:")
    print(f"C: {best['C']}")
    print(f"class_weight: {best['class_weight']}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")
    print(f"Weighted F1: {best['weighted_f1']:.4f}")

if __name__ == "__main__":
    main()
