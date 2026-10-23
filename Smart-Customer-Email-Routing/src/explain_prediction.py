
from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "linear_svm.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"

def main():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    subject = "Unable to access my account"
    body = (
        "I cannot log in after resetting my password. "
        "The application keeps showing an error message."
    )
    email_text = subject + " " + body

    X = vectorizer.transform([email_text])
    feature_names = vectorizer.get_feature_names_out()
    active_indices = X.indices
    active_values = X.data

    classes = model.classes_
    target_queue = "Billing and Payments"
    target_index = list(classes).index(target_queue)

    # LinearSVC uses one-vs-rest classifiers for multiclass classification.
    weights = model.coef_[target_index]

    contributions = []
    for index, tfidf_value in zip(active_indices, active_values):
        contribution = float(weights[index] * tfidf_value)
        contributions.append(
            (feature_names[index], tfidf_value, contribution)
        )

    contributions.sort(key=lambda item: item[2], reverse=True)

    print("=" * 60)
    print("FEATURE CONTRIBUTIONS TO BILLING AND PAYMENTS")
    print("=" * 60)
    print("Predicted queue:", model.predict(X)[0])
    print("\nTop features supporting Billing and Payments:")
    for word, tfidf, contribution in contributions[:10]:
        print(
            f"{word:25s} TF-IDF={tfidf:.4f} "
            f"contribution={contribution:+.4f}"
        )

    print("\nTop features opposing Billing and Payments:")
    for word, tfidf, contribution in contributions[-10:][::-1]:
        print(
            f"{word:25s} TF-IDF={tfidf:.4f} "
            f"contribution={contribution:+.4f}"
        )

    print(
        "\nNote: these are feature contributions to the "
        "Billing-vs-rest decision score, not probabilities."
    )

if __name__ == "__main__":
    main()
