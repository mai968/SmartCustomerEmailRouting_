
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "linear_svm.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"


def main():
    # 1. Tải mô hình và vectorizer đã lưu
    print("Loading saved model and vectorizer...")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # 2. Tạo một email mẫu chưa dùng trong bước huấn luyện
    # subject = "Unable to access my account"
    # body = (
    #     "I cannot log in after resetting my password. "
    #     "The application keeps showing an error message. "
    #     "Could you please help me fix this issue?"
    # )

    # email_text = subject + " " + body

        
    email_text = [
        {
            "subject": "Unable to access my account",
            "body": (
                "I cannot log in after resetting my password. "
                "The application keeps showing an error message."
            ),
        },
        {
            "subject": "Unable to access the application",
            "body": (
                "The application displays a connection error. "
                "I have restarted my computer, but the issue persists."
            ),
        },
        {
            "subject": "Incorrect invoice amount",
            "body": (
                "The amount charged on my invoice is incorrect. "
                "Please review the payment and issue a correction."
            ),
        },
    ]

    for email in email_text:
        text = email["subject"] + " " + email["body"]
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]

        print(f"\nSubject: {email['subject']}")
        print(f"Predicted queue: {prediction}")


    # 3. Chuyển email thành đặc trưng bằng vectorizer đã lưu
    email_text = email["subject"] + " " + email["body"]
    email_tfidf = vectorizer.transform([email_text])
    # 4. Dự đoán queue
    predicted_queue = model.predict(email_tfidf)[0]

    # print("\n" + "=" * 45)
    # print("SAVED MODEL TEST")
    # print("=" * 45)
    # print(f"Subject: {subject}")
    print(f"Predicted queue: {predicted_queue}")
    print("Model loaded and prediction completed successfully.")


if __name__ == "__main__":
    main()
