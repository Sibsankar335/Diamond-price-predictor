"""
model_training.py
------------------
Loads cleaned diamond data, splits into train/test, scales numeric
features, trains multiple candidate regression models, picks the best by
cross-validated R^2, and saves the final model + scaler as pickle files
in models/.

Run:
    python src/model_training.py
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from utils import (
    DATA_PATH,
    TEST_DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def get_candidate_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    }


def train_and_select_best(X_train, y_train):
    """Cross-validate each candidate model on R^2 and return the best fitted estimator."""
    best_name, best_score, best_model = None, -float("inf"), None

    for name, model in get_candidate_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        mean_score = scores.mean()
        print(f"{name:20s} CV R^2: {mean_score:.4f} (+/- {scores.std():.4f})")

        if mean_score > best_score:
            best_name, best_score, best_model = name, mean_score, model

    print(f"\nBest model selected: {best_name} (CV R^2: {best_score:.4f})")
    best_model.fit(X_train, y_train)
    return best_name, best_model


def main():
    df = load_clean_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (fit on train only, to avoid leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_name, best_model = train_and_select_best(X_train_scaled, y_train)

    # Save model + scaler
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Persist the exact test split so model_evaluation.py reuses the same rows
    X_test.assign(price=y_test.values).to_csv(TEST_DATA_PATH, index=False)

    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")


if __name__ == "__main__":
    main()
