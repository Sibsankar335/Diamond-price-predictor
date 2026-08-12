# 💎 Diamond Price Predictor

An end-to-end ML project that predicts the market **price of a diamond**
from its 4 Cs (carat, cut, color, clarity) and physical dimensions, using
the classic **Diamonds dataset** (~54,000 real diamond sales records).

**Test R²: 0.982** | **MAE: ~$280** | **MAPE: ~8%** (Gradient Boosting Regressor)

## Project Structure

```
diamond_price_predictor/
├── data/
│   ├── diamonds_raw.csv        # raw data (54k diamonds)
│   ├── diamonds_clean.csv      # cleaned + encoded dataset used for training
│   └── diamonds_test.csv       # held-out test split (created by training)
├── models/
│   ├── diamond_model.pkl       # trained regressor (pickle)
│   ├── scaler.pkl              # fitted StandardScaler (pickle)
│   └── metrics.json            # evaluation metrics (created by evaluation script)
├── src/
│   ├── __init__.py
│   ├── utils.py                 # shared paths/constants
│   ├── data_cleaning.py         # Step 1: load, clean, encode data
│   ├── model_training.py        # Step 2: train + select best model
│   └── model_evaluation.py      # Step 3: evaluate on test set
├── app.py                       # Streamlit interactive prediction app
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (in order)
```bash
cd src
python data_cleaning.py       # cleans raw data -> data/diamonds_clean.csv
python model_training.py      # trains models, saves best one -> models/diamond_model.pkl
python model_evaluation.py    # evaluates on test set -> models/metrics.json
cd ..
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```
This opens an interactive UI where you can:
- **Predict Price tab**: set carat, cut, color, clarity, and dimensions to
  get a live price estimate, plus a list of similar real diamonds for
  comparison.
- **Model Performance tab**: view R², MAE, RMSE, MAPE, and feature
  importance.
- **Data tab**: explore price vs. carat scatter plots and average price
  by cut quality.

## Dataset

The [Diamonds dataset](https://ggplot2.tidyverse.org/reference/diamonds.html)
contains prices and attributes of ~54,000 round-cut diamonds:

| Feature | Description |
|---|---|
| `carat` | Weight of the diamond (0.2 – 5.01) |
| `cut` | Quality: Fair, Good, Very Good, Premium, Ideal |
| `color` | Diamond color, D (best) to J (worst) |
| `clarity` | Clarity grade: I1 (worst) to IF (flawless) |
| `depth` | Total depth % = z / mean(x, y) |
| `table` | Width of top facet relative to widest point |
| `x`, `y`, `z` | Length, width, depth in mm |
| `price` | Price in US dollars (target) |

Categorical quality grades (`cut`, `color`, `clarity`) are encoded as
ordered integer ranks in `data_cleaning.py` so models can exploit their
natural ordering (e.g., Ideal > Premium > Very Good > Good > Fair).

## Model Selection

`model_training.py` cross-validates three candidates (Linear Regression,
Random Forest, Gradient Boosting) and automatically keeps the one with the
best 5-fold CV R². Gradient Boosting was selected in the default run,
narrowly ahead of Random Forest — both dramatically outperform plain
linear regression, since price depends non-linearly on carat.

## Notes

- Feature scaling (`StandardScaler`) is fit only on the training split to
  avoid data leakage, then reused at inference time via `scaler.pkl`.
- Outlier removal in `data_cleaning.py` uses a wide IQR bound (3x) to
  drop only extreme/erroneous rows while preserving legitimate
  large/expensive diamonds.
- Re-running `model_training.py` regenerates `data/diamonds_test.csv`,
  which `model_evaluation.py` depends on — always run training before

  # 💎 Diamond Price Predictor

🔗 **Live Demo:**[ https://sibsankar335-diamond-price-predictor-app-pedcm.streamlit.app/](https://sibsankar335-diamond-price-predictor-app-pe3dcm.streamlit.app/)

An end-to-end ML project that predicts the market **price of a diamond**
from its 4 Cs (carat, cut, color, clarity) and physical dimensions, using
the classic **Diamonds dataset** (~54,000 real diamond sales records).

**Test R²: 0.982** | **MAE: ~$280** | **MAPE: ~8%** (Gradient Boosting Regressor)
  evaluation.

