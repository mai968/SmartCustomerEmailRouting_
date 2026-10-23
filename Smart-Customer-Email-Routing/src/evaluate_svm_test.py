
from pathlib import Path
import time
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def main():
    print("=" * 65)
    print("FINAL TEST EVALUATION - LINEAR SVM")
    print("=" * 65)

    # 1. Load training and test data
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}")

    # 2. Convert email text into TF-IDF features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=100000,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train["text"].fillna(""))
    X_test = vectorizer.transform(test["text"].fillna(""))

    print(f"Actual TF-IDF features: {X_train.shape[1]}")

    # 3. Train the selected model
    model = LinearSVC(
        C=10.0,
        random_state=42,
        max_iter=5000,
    )

    start_time = time.time()
    model.fit(X_train, train["queue"])
    training_time = time.time() - start_time

    # 4. Evaluate on the test set
    y_true = test["queue"]
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(
        y_true, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, predictions, average="weighted", zero_division=0
    )

    print("\n" + "=" * 65)
    print("FINAL TEST PERFORMANCE")
    print("=" * 65)
    print(f"Accuracy:   {accuracy:.4f} ({accuracy:.2%})")
    print(f"Macro F1:   {macro_f1:.4f}")
    print(f"Weighted F1:{weighted_f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    # 5. Save classification report
    report = classification_report(
        y_true,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        REPORT_DIR / "svm_test_classification_report.csv"
    )

    # 6. Save confusion matrix
    labels = model.classes_
    cm = confusion_matrix(y_true, predictions, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels],
    )
    cm_df.to_csv(REPORT_DIR / "svm_test_confusion_matrix.csv")

    # 7. Save individual predictions
    predictions_df = pd.DataFrame({
        "actual_queue": y_true.to_numpy(),
        "predicted_queue": predictions,
        "correct": y_true.to_numpy() == predictions,
        "text": test["text"].fillna("").to_numpy(),
    })
    predictions_df.to_csv(
        REPORT_DIR / "svm_test_predictions.csv",
        index=False,
    )

    # 8. Save overall summary
    summary = pd.DataFrame([{
        "model": "Linear SVM",
        "C": 10.0,
        "ngram_range": "(1, 2)",
        "max_features_limit": 100000,
        "actual_features": X_train.shape[1],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_time_seconds": round(training_time, 2),
    }])
    summary.to_csv(
        REPORT_DIR / "svm_test_summary.csv",
        index=False,
    )

    print("\nReports saved:")
    print(REPORT_DIR / "svm_test_classification_report.csv")
    print(REPORT_DIR / "svm_test_confusion_matrix.csv")
    print(REPORT_DIR / "svm_test_predictions.csv")
    print(REPORT_DIR / "svm_test_summary.csv")

    print("\nTest evaluation completed.")


if __name__ == "__main__":
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    main()
