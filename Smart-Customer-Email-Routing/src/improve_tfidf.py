
# from pathlib import Path
# import time

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score, f1_score


# # 1. Xac dinh thu muc du an
# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = BASE_DIR / "data" / "processed"
# REPORT_DIR = BASE_DIR / "reports"

# TRAIN_PATH = DATA_DIR / "train.csv"
# VAL_PATH = DATA_DIR / "val.csv"


# def main():
#     print("=" * 65)
#     print("SMART CUSTOMER EMAIL ROUTING - TF-IDF EXPERIMENTS")
#     print("=" * 65)

#     # 2. Doc du lieu train va validation
#     print("\nLoading data...")
#     train = pd.read_csv(TRAIN_PATH)
#     val = pd.read_csv(VAL_PATH)

#     print(f"Training samples:   {len(train):,}")
#     print(f"Validation samples: {len(val):,}")

#     # Kiem tra cac cot can thiet
#     required_columns = {"text", "queue"}
#     if not required_columns.issubset(train.columns):
#         raise ValueError("train.csv must contain 'text' and 'queue' columns")
#     if not required_columns.issubset(val.columns):
#         raise ValueError("val.csv must contain 'text' and 'queue' columns")

#     # 3. Thu 3 cau hinh TF-IDF
#     experiments = [
#         {
#             "name": "Baseline",
#             "ngram_range": (1, 2),
#             "max_features": 50000,
#         },
#         {
#             "name": "Trigram",
#             "ngram_range": (1, 3),
#             "max_features": 50000,
#         },
#         {
#             "name": "More features",
#             "ngram_range": (1, 2),
#             "max_features": 100000,
#         },
#     ]

#     results = []

#     # 4. Chay tung thi nghiem
#     for config in experiments:
#         print("\n" + "-" * 65)
#         print(f"Experiment: {config['name']}")
#         print(f"N-grams: {config['ngram_range']}")
#         print(f"Max features: {config['max_features']:,}")

#         start_time = time.time()

#         # Tao bo tu vung va chuyen email thanh vector
#         vectorizer = TfidfVectorizer(
#             lowercase=True,
#             ngram_range=config["ngram_range"],
#             max_features=config["max_features"],
#             sublinear_tf=True,
#         )

#         X_train = vectorizer.fit_transform(train["text"].fillna(""))
#         X_val = vectorizer.transform(val["text"].fillna(""))

#         print(f"Actual TF-IDF features: {X_train.shape[1]:,}")

#         # Huan luyen Linear SVM
#         model = LinearSVC(
#             C=10.0,
#             random_state=42,
#             max_iter=5000,
#         )
#         model.fit(X_train, train["queue"])

#         # Du doan queue cho tap validation
#         predictions = model.predict(X_val)

#         accuracy = accuracy_score(val["queue"], predictions)
#         macro_f1 = f1_score(
#             val["queue"], predictions, average="macro"
#         )
#         weighted_f1 = f1_score(
#             val["queue"], predictions, average="weighted"
#         )

#         elapsed = time.time() - start_time

#         results.append({
#             "experiment": config["name"],
#             "ngram_range": str(config["ngram_range"]),
#             "max_features_limit": config["max_features"],
#             "actual_features": X_train.shape[1],
#             "accuracy": accuracy,
#             "macro_f1": macro_f1,
#             "weighted_f1": weighted_f1,
#             "time_seconds": round(elapsed, 2),
#         })

#         print(f"Accuracy:    {accuracy:.4f}")
#         print(f"Macro F1:    {macro_f1:.4f}")
#         print(f"Weighted F1: {weighted_f1:.4f}")
#         print(f"Time:        {elapsed:.1f} seconds")

#     # 5. Luu ket qua vao reports
#     REPORT_DIR.mkdir(parents=True, exist_ok=True)

#     results_df = pd.DataFrame(results)
#     results_df = results_df.sort_values(
#         "macro_f1", ascending=False
#     )

#     output_path = REPORT_DIR / "tfidf_experiments.csv"
#     results_df.to_csv(output_path, index=False)

#     print("\n" + "=" * 65)
#     print("FINAL COMPARISON (sorted by Macro F1)")
#     print("=" * 65)
#     print(results_df.to_string(index=False))
#     print(f"\nResults saved to: {output_path}")


# if __name__ == "__main__":
#     main()
