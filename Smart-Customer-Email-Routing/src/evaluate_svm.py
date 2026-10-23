
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("SMART CUSTOMER EMAIL ROUTING - SVM EVALUATION")
print("=" * 70)

print(f"Project directory: {BASE_DIR}")
print(f"Data directory:    {DATA_DIR}")
print(f"Report directory:  {REPORT_DIR}")


# ============================================================
# 2. LOAD TRAINING AND VALIDATION DATA
# ============================================================

train_path = DATA_DIR / "train.csv"
val_path = DATA_DIR / "val.csv"

if not train_path.exists():
    raise FileNotFoundError(f"Training file not found: {train_path}")

if not val_path.exists():
    raise FileNotFoundError(f"Validation file not found: {val_path}")

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

required_columns = {"text", "queue"}

for file_name, df in [
    ("train.csv", train_df),
    ("val.csv", val_df),
]:
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{file_name} is missing columns: {missing_columns}"
        )

# Remove rows with missing target labels, if any
train_df = train_df.dropna(subset=["queue"]).copy()
val_df = val_df.dropna(subset=["queue"]).copy()

X_train_text = train_df["text"].fillna("").astype(str)
y_train = train_df["queue"].astype(str)

X_val_text = val_df["text"].fillna("").astype(str)
y_val = val_df["queue"].astype(str)

print("\nDATA SUMMARY")
print("-" * 70)
print(f"Training samples:   {len(train_df):,}")
print(f"Validation samples: {len(val_df):,}")
print(f"Number of queues:   {y_train.nunique()}")


# ============================================================
# 3. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\nFitting TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),   #xem xét từ đơn và cụm hai từ, ví dụ problem và login problem.
    max_features=50000,   #giới hạn số đặc trưng văn bản tối đa ở 50.000.
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

print(f"Number of TF-IDF features: {X_train.shape[1]:,}")


# ============================================================
# 4. TRAIN LINEAR SVM
# ============================================================

print("\nTraining Linear SVM...")
print("Configuration: C=10.0, class_weight=None")

model = LinearSVC(
    C=10.0,
    random_state=42,
    max_iter=5000,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("Training and prediction completed.")


# ============================================================
# 5. OVERALL MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0,
)
weighted_f1 = f1_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0,
)

print("\nOVERALL PERFORMANCE")
print("-" * 70)
print(f"Accuracy:   {accuracy:.4f} ({accuracy:.2%})")
print(f"Macro F1:   {macro_f1:.4f}")
print(f"Weighted F1:{weighted_f1:.4f}")


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

report_text = classification_report(
    y_val,
    y_pred,
    labels=sorted(y_train.unique()),
    zero_division=0,
)

print("\nCLASSIFICATION REPORT")
print("=" * 70)
print(report_text)

report_dict = classification_report(
    y_val,
    y_pred,
    labels=sorted(y_train.unique()),
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = REPORT_DIR / "svm_classification_report.csv"
report_df.to_csv(report_path, index=True)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

labels = sorted(y_train.unique())

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual: {label}" for label in labels],
    columns=[f"Predicted: {label}" for label in labels],
)

cm_path = REPORT_DIR / "svm_confusion_matrix.csv"
cm_df.to_csv(cm_path, index=True)

print("\nCONFUSION MATRIX")
print("-" * 70)
print("Rows = actual queues; columns = predicted queues")
print(cm_df.to_string())


# ============================================================
# 8. FIND THE MOST FREQUENT MISCLASSIFICATIONS
# ============================================================

cm_errors = cm.copy()

# Ignore correct predictions on the diagonal
np.fill_diagonal(cm_errors, 0)

errors = []

for i, actual_label in enumerate(labels):
    for j, predicted_label in enumerate(labels):
        error_count = int(cm_errors[i, j])

        if error_count > 0:
            errors.append({
                "actual_queue": actual_label,
                "predicted_queue": predicted_label,
                "error_count": error_count,
            })

errors_df = pd.DataFrame(
    errors,
    columns=[
        "actual_queue",
        "predicted_queue",
        "error_count",
    ],
)

if not errors_df.empty:
    errors_df = errors_df.sort_values(
        by="error_count",
        ascending=False,
    ).reset_index(drop=True)

    errors_df.insert(
        0,
        "rank",
        range(1, len(errors_df) + 1),
    )

    print("\nTOP 15 MISCLASSIFIED QUEUE PAIRS")
    print("=" * 70)
    print(
        errors_df.head(15).to_string(index=False)
    )

else:
    print("\nNo misclassifications found.")

errors_path = REPORT_DIR / "svm_top_misclassifications.csv"
errors_df.to_csv(errors_path, index=False)


# ============================================================
# 9. SAVE A SUMMARY OF THE RESULTS
# ============================================================

summary_df = pd.DataFrame([{
    "model": "Linear SVM",
    "C": 10.0,
    "class_weight": "None",
    "tfidf_max_features": 50000,
    "tfidf_ngram_range": "(1, 2)",
    "train_samples": len(train_df),
    "validation_samples": len(val_df),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
}])

summary_path = REPORT_DIR / "svm_evaluation_summary.csv"
summary_df.to_csv(summary_path, index=False)


# ============================================================
# 10. FINISH
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print("\nFiles saved:")
print(f"1. {report_path}")
print(f"2. {cm_path}")
print(f"3. {errors_path}")
print(f"4. {summary_path}")
