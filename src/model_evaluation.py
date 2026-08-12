"""
model_evaluation.py
--------------------
Loads the saved model + scaler + held-out test split, computes regression
metrics (R^2, MAE, RMSE, MAPE), and saves them to models/metrics.json for
the Streamlit app to display.

Run:
    python src/model_evaluation.py
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils import MODEL_PATH, SCALER_PATH, METRICS_PATH, TEST_DATA_PATH, FEATURE_COLUMNS


def main():
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["price"]

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    print(f"R^2   : {r2:.4f}")
    print(f"MAE   : ${mae:,.2f}")
    print(f"RMSE  : ${rmse:,.2f}")
    print(f"MAPE  : {mape:.2f}%")

    # Feature importance (if the model supports it)
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        feature_importance = dict(
            sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    metrics = {
        "r2_score": r2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "model_type": type(model).__name__,
        "feature_importance": feature_importance,
        "n_test_samples": len(y_test),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
