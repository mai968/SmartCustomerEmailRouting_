
from pathlib import Path
import time

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# 1. Duong dan
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def main():
    # 2. Doc du lieu
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    X_train_text = train_df["text"].fillna("")
    y_train = train_df["queue"]

    X_val_text = val_df["text"].fillna("")
    y_val = val_df["queue"]

    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))

    # 3. Chuyen email thanh TF-IDF
    print("\nCreating TF-IDF features...")

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True
    )

    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)

    print("TF-IDF train shape:", X_train_tfidf.shape)
    print("TF-IDF validation shape:", X_val_tfidf.shape)

    # 4. SVD cho Gradient Boosting va MLP
    # Giam so chieu de giam RAM va thoi gian tinh toan.
    print("\nCreating reduced features for Gradient Boosting and MLP...")

    svd = TruncatedSVD(
        n_components=100,
        random_state=RANDOM_STATE
    )

    X_train_reduced = svd.fit_transform(X_train_tfidf)
    X_val_reduced = svd.transform(X_val_tfidf)

    print(
        f"Explained variance ratio sum: "
        f"{svd.explained_variance_ratio_.sum():.4f}"
    )

    # 5. Khai bao 5 model
    models = {
        "KNN": (
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="cosine",
                algorithm="brute",
                n_jobs=-1
            ),
            X_train_tfidf,
            X_val_tfidf
        ),

        "Decision Tree": (
            DecisionTreeClassifier(
                max_depth=50,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            X_train_tfidf,
            X_val_tfidf
        ),

        "Random Forest": (
            RandomForestClassifier(
                n_estimators=100,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ),
            X_train_tfidf,
            X_val_tfidf
        ),

        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=2,
                random_state=RANDOM_STATE
            ),
            X_train_reduced,
            X_val_reduced
        ),

        "MLPClassifier": (
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(128,),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    batch_size=128,
                    learning_rate_init=0.001,
                    max_iter=30,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    random_state=RANDOM_STATE,
                    verbose=False
                )
            ),
            X_train_reduced,
            X_val_reduced
        )
    }

    new_results = []

    # 6. Train va danh gia
    for name, (model, Xtr, Xv) in models.items():
        print("\n" + "=" * 60)
        print("Training:", name)

        start_time = time.time()

        try:
            model.fit(Xtr, y_train)
            predictions = model.predict(Xv)

            accuracy = accuracy_score(y_val, predictions)
            macro_f1 = f1_score(
                y_val, predictions,
                average="macro",
                zero_division=0
            )
            weighted_f1 = f1_score(
                y_val, predictions,
                average="weighted",
                zero_division=0
            )

            elapsed = time.time() - start_time

            print(f"Accuracy:    {accuracy:.4f}")
            print(f"Macro F1:    {macro_f1:.4f}")
            print(f"Weighted F1: {weighted_f1:.4f}")
            print(f"Training + prediction time: {elapsed:.1f} seconds")

            print("\nClassification Report:")
            print(
                classification_report(
                    y_val,
                    predictions,
                    zero_division=0
                )
            )

            new_results.append({
                "Model": name,
                "Accuracy": accuracy,
                "Macro_F1": macro_f1,
                "Weighted_F1": weighted_f1,
                "Time_seconds": round(elapsed, 1)
            })

        except Exception as error:
            print(f"ERROR while running {name}: {error}")

    # 7. Luu ket qua 5 model moi
    new_df = pd.DataFrame(new_results)
    new_path = REPORT_DIR / "extended_model_comparison.csv"
    new_df.to_csv(new_path, index=False, encoding="utf-8-sig")

    # 8. Gop voi ket qua 3 model truoc neu co
    old_path = REPORT_DIR / "model_comparison.csv"

    if old_path.exists() and not new_df.empty:
        old_df = pd.read_csv(old_path)

        combined_df = pd.concat(
            [old_df, new_df],
            ignore_index=True,
            sort=False
        )

        # Bo dong trung ten model neu chay lai script
        combined_df = combined_df.drop_duplicates(
            subset=["Model"],
            keep="last"
        )

        combined_df = combined_df.sort_values(
            by="Macro_F1",
            ascending=False
        )

        combined_path = REPORT_DIR / "all_models_comparison.csv"
        combined_df.to_csv(
            combined_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n" + "=" * 60)
        print("ALL MODELS - SORTED BY MACRO F1")
        print(combined_df.to_string(index=False))
        print(f"\nSaved all results to: {combined_path}")

    print(f"\nSaved new model results to: {new_path}")


if __name__ == "__main__":
    main()
