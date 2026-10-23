
# from pathlib import Path
# import pandas as pd
# import joblib

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix
# )

# PROJECT_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = PROJECT_DIR / "data" / "processed"
# MODEL_DIR = PROJECT_DIR / "models"
# MODEL_DIR.mkdir(parents=True, exist_ok=True)


# def main():
#     train_df = pd.read_csv(DATA_DIR / "train.csv")
#     test_df = pd.read_csv(DATA_DIR / "test.csv")

#     X_train = train_df["text"].fillna("")
#     y_train = train_df["queue"]

#     X_test = test_df["text"].fillna("")
#     y_test = test_df["queue"]

#     # Tao quy trinh: chuyen text thanh TF-IDF va phan loai
#     model = Pipeline([
#         (
#             "tfidf",
#             TfidfVectorizer(
#                 lowercase=True,
#                 ngram_range=(1, 2),
#                 max_features=50000,
#                 sublinear_tf=True
#             )
#         ),
#         (
#             "classifier",
#             LogisticRegression(
#                 max_iter=1000,
#                 class_weight="balanced"
#             )
#         )
#     ])

#     print("Dang huan luyen mo hinh...")
#     model.fit(X_train, y_train)

#     print("\nDang danh gia tren test set...")
#     predictions = model.predict(X_test)

#     print("\n=== ACCURACY ===")
#     print(round(accuracy_score(y_test, predictions), 4))

#     print("\n=== CLASSIFICATION REPORT ===")
#     print(classification_report(
#         y_test,
#         predictions,
#         zero_division=0
#     ))

#     print("\n=== CONFUSION MATRIX ===")
#     labels = model.named_steps["classifier"].classes_
#     cm = confusion_matrix(y_test, predictions, labels=labels)
#     print("Thu tu nhan:")
#     print(list(labels))
#     print(cm)

#     model_path = MODEL_DIR / "email_routing_model.joblib"
#     joblib.dump(model, model_path)

#     print("\nDa luu mo hinh tai:", model_path)


# if __name__ == "__main__":
#     main()
