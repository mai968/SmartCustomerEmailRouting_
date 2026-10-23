
from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# 1. Xác định thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "linear_svm.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"


def main():
    # 2. Đọc dữ liệu huấn luyện
    print("Loading training data...")

    train_df = pd.read_csv(TRAIN_PATH)

    if not {"text", "queue"}.issubset(train_df.columns):
        raise ValueError(
            "train.csv must contain 'text' and 'queue' columns."
        )

    X_train = train_df["text"].fillna("").astype(str)
    y_train = train_df["queue"]

    print(f"Training samples: {len(train_df)}")
    print(f"Number of queues: {y_train.nunique()}")

    # 3. Chuyển văn bản thành đặc trưng TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=100000,
        sublinear_tf=True,
    )

    print("\nTransforming text with TF-IDF...")
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # 4. Khởi tạo và huấn luyện Linear SVM
    model = LinearSVC(
        C=10.0,
        random_state=42,
        max_iter=5000,
    )

    print("Training Linear SVM...")
    start_time = time.time()

    model.fit(X_train_tfidf, y_train)

    elapsed = time.time() - start_time

    # 5. Tạo thư mục models nếu chưa tồn tại
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 6. Lưu mô hình và vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # 7. Hiển thị kết quả
    print("\n" + "=" * 50)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 50)

    print(f"Training time: {elapsed:.2f} seconds")
    print(
        "TF-IDF features:",
        len(vectorizer.get_feature_names_out()),
    )
    print(f"Model path: {MODEL_PATH}")
    print(f"Vectorizer path: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
