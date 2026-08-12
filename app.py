"""
app.py
------
Streamlit app for interactive Diamond Price prediction.
Loads the trained model + scaler from models/ and lets the user input
diamond characteristics (carat, cut, color, clarity, dimensions) to get
a live price prediction.

Run:
    streamlit run app.py
"""

import json
import pickle
import pandas as pd
import streamlit as st

from src.utils import (
    MODEL_PATH,
    SCALER_PATH,
    METRICS_PATH,
    DATA_PATH,
    FEATURE_COLUMNS,
    CUT_ORDER,
    COLOR_ORDER,
    CLARITY_ORDER,
)

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="wide")


@st.cache_resource
def load_model_and_scaler():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


model, scaler = load_model_and_scaler()
df = load_reference_data()
metrics = load_metrics()

# Ordered lists (best-looking-first) for select boxes
CUT_OPTIONS = sorted(CUT_ORDER, key=CUT_ORDER.get, reverse=True)
COLOR_OPTIONS = sorted(COLOR_ORDER, key=COLOR_ORDER.get, reverse=True)
CLARITY_OPTIONS = sorted(CLARITY_ORDER, key=CLARITY_ORDER.get, reverse=True)

st.title("💎 Diamond Price Predictor")
st.write(
    "Estimate the market price of a diamond from its **4 Cs** (carat, cut, "
    "color, clarity) and physical dimensions. Model trained on ~54,000 "
    "real diamond sales records."
)

tab_predict, tab_metrics, tab_data = st.tabs(["🔮 Predict Price", "📊 Model Performance", "📁 Data"])

# ---------------------------------------------------------------------------
# TAB 1: Prediction
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter diamond characteristics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**The 4 Cs**")
        carat = st.slider(
            "Carat (weight)", min_value=0.2, max_value=float(df["carat"].max()),
            value=1.0, step=0.01,
        )
        cut = st.selectbox("Cut quality", CUT_OPTIONS, index=0)
        color = st.selectbox("Color grade (D = colorless/best, J = more color)", COLOR_OPTIONS, index=3)
        clarity = st.selectbox("Clarity grade (IF = flawless/best, I1 = included)", CLARITY_OPTIONS, index=3)

    with col2:
        st.markdown("**Physical dimensions (mm)**")
        x = st.slider("Length (x)", min_value=3.0, max_value=float(df["x"].max()), value=6.5, step=0.01)
        y = st.slider("Width (y)", min_value=3.0, max_value=float(df["y"].max()), value=6.5, step=0.01)
        z = st.slider("Depth (z)", min_value=2.0, max_value=float(df["z"].max()), value=4.0, step=0.01)
        depth = st.slider(
            "Depth %", min_value=float(df["depth"].min()), max_value=float(df["depth"].max()),
            value=61.5, step=0.1,
            help="Depth = z / mean(x, y) as a percentage — affects how a diamond sparkles.",
        )
        table = st.slider(
            "Table %", min_value=float(df["table"].min()), max_value=float(df["table"].max()),
            value=57.0, step=0.5,
            help="Table = width of the flat top facet as a percentage of overall diameter.",
        )

    st.markdown("---")

    if st.button("Predict Price", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "carat": carat,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
            "cut_encoded": CUT_ORDER[cut],
            "color_encoded": COLOR_ORDER[color],
            "clarity_encoded": CLARITY_ORDER[clarity],
        }])[FEATURE_COLUMNS]

        input_scaled = scaler.transform(input_df)
        predicted_price = model.predict(input_scaled)[0]

        st.success(f"### Estimated Price: ${predicted_price:,.2f}")
        st.caption(
            f"Typical model error on similar diamonds is about "
            f"±${metrics['mae']:,.0f} ({metrics['mape']:.1f}% MAPE)."
        )

        # Show comparable diamonds from the dataset
        similar = df[
            (df["carat"].between(carat - 0.1, carat + 0.1))
            & (df["cut_encoded"] == CUT_ORDER[cut])
        ].sort_values("carat")

        if len(similar) > 0:
            st.write(f"**Similar diamonds in the dataset** (±0.1 carat, same cut):")
            st.dataframe(
                similar[["carat", "cut", "color", "clarity", "price"]].head(10),
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# TAB 2: Model performance
# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Model Performance")
    st.write(f"**Model type:** {metrics['model_type']}")
    st.write(f"**Test samples:** {metrics['n_test_samples']:,}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{metrics['r2_score']:.4f}")
    m2.metric("MAE", f"${metrics['mae']:,.2f}")
    m3.metric("RMSE", f"${metrics['rmse']:,.2f}")
    m4.metric("MAPE", f"{metrics['mape']:.2f}%")

    if metrics.get("feature_importance"):
        st.write("**Feature importance:**")
        importance_df = pd.DataFrame(
            list(metrics["feature_importance"].items()),
            columns=["Feature", "Importance"],
        ).set_index("Feature")
        st.bar_chart(importance_df)

# ---------------------------------------------------------------------------
# TAB 3: Data explorer
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset preview")
    st.write(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    st.dataframe(df.head(20), use_container_width=True)

    st.write("**Price vs. Carat:**")
    st.scatter_chart(df.sample(min(2000, len(df))), x="carat", y="price")

    st.write("**Average price by cut quality:**")
    st.bar_chart(df.groupby("cut", observed=True)["price"].mean())
