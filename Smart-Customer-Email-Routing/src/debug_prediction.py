
from pathlib import Path
import joblib

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

    # Convert email to TF-IDF features
    features = vectorizer.transform([email_text])
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = features.nonzero()[1]

    print("=" * 55)
    print("DEBUGGING ONE EMAIL PREDICTION")
    print("=" * 55)

    print("\nPredicted queue:", model.predict(features)[0])

    print("\nWords/phrases represented in TF-IDF:")
    if len(nonzero_indices) == 0:
        print("No vocabulary features matched.")
    else:
        matched = sorted(
            [
                (feature_names[i], float(features[0, i]))
                for i in nonzero_indices
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        for word, score in matched:
            print(f"{word:25s} {score:.4f}")

    # LinearSVC decision scores are relative margins, not probabilities
    scores = model.decision_function(features).ravel()
    ranking = scores.argsort()[::-1]

    print("\nTop 5 queue scores (not probabilities):")
    for i in ranking[:5]:
        print(f"{model.classes_[i]:40s} {scores[i]:.4f}")

if __name__ == "__main__":
    main()
