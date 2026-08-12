"""
utils.py
--------
Small shared helpers/constants used across training, evaluation, and the
Streamlit app. Centralizing these avoids repeating paths/column lists in
every file.
"""

import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "diamonds_clean.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "diamonds_test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "diamond_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# Numeric + encoded categorical features used for training
FEATURE_COLUMNS = [
    "carat",
    "depth",
    "table",
    "x",
    "y",
    "z",
    "cut_encoded",
    "color_encoded",
    "clarity_encoded",
]

TARGET_COLUMN = "price"

# Ordered quality grades (worst -> best), used to build UI dropdowns
CUT_ORDER = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
COLOR_ORDER = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
CLARITY_ORDER = {
    "I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8,
}
