
# from pathlib import Path
# import time

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     classification_report,
#     confusion_matrix,
# )


# # 1. Xac dinh duong dan du an
# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = BASE_DIR / "data" / "processed"
# REPORT_DIR = BASE_DIR / "reports"

# TRAIN_PATH = DATA_DIR / "train.csv"
# VAL_PATH = DATA_DIR / "val.csv"


# def main():
#     print("=" * 70)
#     print("SMART CUSTOMER EMAIL ROUTING - IMPROVED SVM EVALUATION")
#     print("=" * 70)

#     # 2. Doc du lieu
#     print("\nLoading training and validation data...")

#     train = pd.read_csv(TRAIN_PATH)
#     val = pd.read_csv(VAL_PATH)

#     print(f"Training samples:   {len(train):,}")
#     print(f"Validation samples: {len(val):,}")
#     print(f"Number of queues:   {train['queue'].nunique()}")

#     # 3. Chuyen email thanh vector TF-IDF
#     print("\nFitting improved TF-IDF vectorizer...")

#     vectorizer = TfidfVectorizer(
#         lowercase=True,
#         ngram_range=(1, 2),
#         max_features=100000,
#         sublinear_tf=True,
#     )

#     X_train = vectorizer.fit_transform(
#         train["text"].fillna("")
#     )
#     X_val = vectorizer.transform(
#         val["text"].fillna("")
#     )

#     print(f"Number of TF-IDF features: {X_train.shape[1]:,}")

#     # 4. Huan luyen Linear SVM
#     print("\nTraining Linear SVM...")

#     model = LinearSVC(
#         C=10.0,
#         random_state=42,
#         max_iter=5000,
#     )

#     start_time = time.time()
#     model.fit(X_train, train["queue"])
#     training_time = time.time() - start_time

#     # 5. Du doan queue tren validation
#     print("\nPredicting validation queues...")

#     predictions = model.predict(X_val)
#     y_true = val["queue"]

#     accuracy = accuracy_score(y_true, predictions)
#     macro_f1 = f1_score(
#         y_true, predictions, average="macro"
#     )
#     weighted_f1 = f1_score(
#         y_true, predictions, average="weighted"
#     )

#     print("\nOVERALL PERFORMANCE")
#     print("-" * 70)
#     print(f"Accuracy:    {accuracy:.4f} ({accuracy:.2%})")
#     print(f"Macro F1:    {macro_f1:.4f}")
#     print(f"Weighted F1: {weighted_f1:.4f}")
#     print(f"Training time: {training_time:.2f} seconds")

#     # 6. Tao bao cao cho tung queue
#     report = classification_report(
#         y_true,
#         predictions,
#         output_dict=True,
#         zero_division=0,
#     )

#     report_df = pd.DataFrame(report).transpose()
#     report_path = REPORT_DIR / "svm_improved_classification_report.csv"
#     report_df.to_csv(report_path, index=True)

#     # 7. Tao confusion matrix
#     labels = model.classes_

#     cm = confusion_matrix(
#         y_true,
#         predictions,
#         labels=labels,
#     )

#     cm_df = pd.DataFrame(
#         cm,
#         index=[f"Actual: {label}" for label in labels],
#         columns=[f"Predicted: {label}" for label in labels],
#     )

#     cm_path = REPORT_DIR / "svm_improved_confusion_matrix.csv"
#     cm_df.to_csv(cm_path, index=True)

#     # 8. Luu tung email va ket qua du doan
#     predictions_df = pd.DataFrame({
#         "actual_queue": y_true.to_numpy(),
#         "predicted_queue": predictions,
#         "correct": y_true.to_numpy() == predictions,
#         "text": val["text"].fillna("").to_numpy(),
#     })

#     predictions_path = REPORT_DIR / "svm_improved_predictions.csv"
#     predictions_df.to_csv(predictions_path, index=False)

#     # 9. Luu bang tong hop metric
#     summary_df = pd.DataFrame([{
#         "model": "Linear SVM",
#         "C": 10.0,
#         "ngram_range": "(1, 2)",
#         "max_features_limit": 100000,
#         "actual_features": X_train.shape[1],
#         "accuracy": accuracy,
#         "macro_f1": macro_f1,
#         "weighted_f1": weighted_f1,
#         "training_time_seconds": round(training_time, 2),
#     }])

#     summary_path = REPORT_DIR / "svm_improved_summary.csv"
#     summary_df.to_csv(summary_path, index=False)

#     # 10. In ra cac queue kho phan loai
#     per_queue = report_df.loc[
#         [label for label in labels if label in report_df.index],
#         ["precision", "recall", "f1-score", "support"],
#     ].sort_values("f1-score")

#     print("\nPERFORMANCE BY QUEUE (lowest F1 first)")
#     print("-" * 70)
#     print(per_queue.to_string())

#     print("\nReports saved:")
#     print(f"1. {report_path}")
#     print(f"2. {cm_path}")
#     print(f"3. {predictions_path}")
#     print(f"4. {summary_path}")

#     print("\nEVALUATION COMPLETED")


# if __name__ == "__main__":
#     REPORT_DIR.mkdir(parents=True, exist_ok=True)
#     main()

