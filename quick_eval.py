import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_dataset() -> pd.DataFrame:
    cleaned_path = Path("Algerian_forest_fires_cleaned_dataset.csv")
    update_path = Path("Algerian_forest_fires_dataset_UPDATE.csv")

    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        # Normalize class labels
        if "Classes" in df.columns:
            df["Classes"] = df["Classes"].astype(str).str.strip().str.lower()
        return df

    if update_path.exists():
        # The UPDATE CSV has some columns with stray spaces; read and clean headers
        df = pd.read_csv(update_path)
        df.columns = [c.strip().replace(" ", "") for c in df.columns]
        # Standardize class column name
        if "Classes" not in df.columns:
            # Try variant names
            for col in df.columns:
                if col.lower().startswith("classes"):
                    df = df.rename(columns={col: "Classes"})
                    break
        df["Classes"] = df["Classes"].astype(str).str.strip().str.lower()
        return df

    raise FileNotFoundError("Could not find dataset CSV in workspace.")


def prepare_features(df: pd.DataFrame):
    if "Classes" not in df.columns:
        raise ValueError("Expected a 'Classes' column in the dataset.")

    # Map classes to binary labels
    class_mapping = {"fire": 1, "not fire": 0}
    y = df["Classes"].map(class_mapping)
    # Drop rows with unknown labels
    mask_valid = y.isin([0, 1])
    df = df.loc[mask_valid].copy()
    y = y.loc[mask_valid].astype(int)

    # Candidate numeric features: drop target only
    X = df.drop(columns=["Classes"], errors="ignore")

    # Keep only numeric columns for modeling
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols].copy()

    return X, y, numeric_cols


def build_models(numeric_features: list[str]):
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    linear_preprocess = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",
    )

    rf_preprocess = ColumnTransformer(
        transformers=[("num", "passthrough", numeric_features)],
        remainder="drop",
    )

    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("pre", linear_preprocess),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Ridge Classifier": Pipeline(
            steps=[
                ("pre", linear_preprocess),
                ("clf", RidgeClassifier(random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("pre", rf_preprocess),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_models():
    df = load_dataset()
    X, y, numeric_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = build_models(numeric_cols)

    results: dict[str, float] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

    # Print concise human-readable summary and a JSON blob for programmatic use
    print("Prediction accuracy (test set):")
    for name, acc in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
        print(f"- {name}: {acc * 100:.2f}%")

    print("\nJSON:")
    print(json.dumps(results))


if __name__ == "__main__":
    evaluate_models()

