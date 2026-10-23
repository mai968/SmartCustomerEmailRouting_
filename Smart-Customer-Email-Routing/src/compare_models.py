
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report


# 1. Xac dinh thu muc project
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 2. Doc train va validation
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    X_train = train_df["text"].fillna("")
    y_train = train_df["queue"]

    X_val = val_df["text"].fillna("")
    y_val = val_df["queue"]

    # 3. Khai bao cac model
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42
        ),
        "Multinomial Naive Bayes": MultinomialNB()
    }

    results = []

    # 4. Train va danh gia tung model
    for name, classifier in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training: {name}")

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=50000,
                    sublinear_tf=True
                )
            ),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            y_val, predictions, average="weighted", zero_division=0
        )

        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Macro F1:   {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_val,
                predictions,
                zero_division=0
            )
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1
        })

    # 5. Luu bang so sanh
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="Macro_F1", ascending=False
    )

    report_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by Macro F1)")
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison to: {report_path}")


if __name__ == "__main__":
    main()
