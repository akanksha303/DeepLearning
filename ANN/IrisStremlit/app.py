"""
🌸 Iris Species Classifier — Streamlit GUI (Prediction-only)
Loads a pre-trained model, scaler, and label encoder — no training happens here.
Expected files in the same folder as this script:
    - model.h5
    - scaler.pkl
    - label_encoder.pkl
    - Iris.csv   (used only to draw charts / slider ranges, not to train)
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as stz 
import plotly.express as px
import plotly.graph_objects as go

from keras.models import load_model

# ────────────────────────────────────────────────────────────────────────────
# SAVED ARTIFACT PATHS
# ────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
DATA_PATH = "Iris.csv"


def _load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        import joblib
        return joblib.load(path)


# ────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* Import modern font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Main App Background */
.stApp {
    background: linear-gradient(
        135deg,
        #f8f7ff 0%,
        #eef3ff 45%,
        #fff5fb 100%
    );
    font-family: 'Inter', sans-serif;
    color: #1f2937;
}


/* Main Header */
.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(
        120deg,
        #5b21b6,
        #7c3aed,
        #c084fc
    );
    border-radius: 22px;
    color: white;
    margin-bottom: 1.8rem;
    box-shadow: 
        0 15px 35px rgba(124,58,237,0.25);
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.4rem;
}

.main-header p {
    font-size: 1.05rem;
    opacity: 0.9;
}


/* Metric Cards */
.metric-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(10px);
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    box-shadow:
        0 8px 25px rgba(0,0,0,0.06);
    border-left: 5px solid #7c3aed;
    margin-bottom: 1rem;
    color: #111827;
}


/* Prediction Result Box */
.prediction-box {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(12px);
    padding: 2rem;
    border-radius: 22px;
    box-shadow:
        0 10px 30px rgba(0,0,0,0.08);
    text-align:center;
    border:1px solid #ddd6fe;
    color:#111827;
}


/* Emoji */
.species-emoji {
    font-size:5rem;
}


/* Sidebar */
section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #1e1b4b,
        #312e81
    );
}


section[data-testid="stSidebar"] * {

    color:#e0e7ff !important;
    font-family:'Inter',sans-serif;
}


/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {

    color:white !important;
}


/* Slider */
div[data-baseweb="slider"] {
    padding-top:0.5rem;
}


/* Tabs */
.stTabs [data-baseweb="tab-list"] {

    gap:12px;
}


.stTabs [data-baseweb="tab"] {

    background:white;
    border-radius:12px;
    padding:12px 22px;
    font-weight:600;
    color:#4c1d95;
}


/* Active tab */
.stTabs [aria-selected="true"] {

    background:#7c3aed !important;
    color:white !important;
}


/* Buttons */
.stButton button {

    background:linear-gradient(
        90deg,
        #7c3aed,
        #a855f7
    );

    color:white;
    border:none;
    border-radius:12px;
    padding:0.6rem 1.5rem;
    font-weight:600;
}


.stButton button:hover {

    background:#5b21b6;
    transform:scale(1.02);

}


/* Dataframe */
[data-testid="stDataFrame"] {

    border-radius:15px;
    overflow:hidden;

}

</style>
""", unsafe_allow_html=True)

SPECIES_INFO = {
    "Iris-setosa": {
        "emoji": "🌸",
        "color": "#7b2ff7",
        "desc": "Smallest petals, easiest to separate from the rest."
    },
    "Iris-versicolor": {
        "emoji": "🌺",
        "color": "#ff6f91",
        "desc": "Mid-sized — sits between setosa and virginica."
    },
    "Iris-virginica": {
        "emoji": "🌷",
        "color": "#3aa6a1",
        "desc": "Largest petals and sepals of the three species."
    },
}
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
FEATURE_LABELS = {
    "SepalLengthCm": "Sepal Length (cm)",
    "SepalWidthCm": "Sepal Width (cm)",
    "PetalLengthCm": "Petal Length (cm)",
    "PetalWidthCm": "Petal Width (cm)",
}


# ────────────────────────────────────────────────────────────────────────────
# LOAD DATA (only for charts & slider ranges — not used for training)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)
    return df


# ────────────────────────────────────────────────────────────────────────────
# LOAD SAVED MODEL / SCALER / ENCODER — no training happens here
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    missing = [p for p in ("model.h5", "scaler.pkl", "label_encoder.pkl") if not os.path.exists(p)]
    if missing:
        st.error(
            "Missing saved file(s): " + ", ".join(missing) +
            ". Make sure model.h5, scaler.pkl, and label_encoder.pkl are in the same folder as app.py."
        )
        st.stop()
    model = load_model("model.h5")
    scaler = _load_pickle("scaler.pkl")
    encoder = _load_pickle("label_encoder.pkl")
    return model, scaler, encoder


df = load_data()
model, scaler, encoder = load_artifacts()

# ────────────────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌸 Iris Species Classifier</h1>
    <p>Live predictions from your pre-trained Neural Network — no training, just inference</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔧 Flower Measurements")
st.sidebar.markdown("Adjust the sliders to describe a flower, then check the **Predict** tab.")

user_input = {}
for feat in FEATURES:
    lo = float(df[feat].min())
    hi = float(df[feat].max())
    default = float(df[feat].mean())
    user_input[feat] = st.sidebar.slider(FEATURE_LABELS[feat], lo, hi, default, step=0.1)

st.sidebar.markdown("---")
st.sidebar.caption("✅ Using pre-trained model.h5 — predictions only, no retraining.")

# ────────────────────────────────────────────────────────────────────────────
# TABS
# ────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Predict", "📊 Explore Data"])

# ---- TAB 1: PREDICT --------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1.3])

    input_df = pd.DataFrame([user_input])[FEATURES]
    input_scaled = scaler.transform(input_df)
    probs = model.predict(input_scaled, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_species = encoder.inverse_transform([pred_idx])[0]
    info = SPECIES_INFO[pred_species]

    with col1:
        st.markdown(f"""
        <div class="prediction-box">
            <div class="species-emoji">{info['emoji']}</div>
            <h2 style="color:{info['color']}; margin-bottom:0;">Iris {pred_species.capitalize()}</h2>
            <p style="color:#666;">{info['desc']}</p>
            <h3 style="color:{info['color']};">Confidence: {probs[pred_idx]*100:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📋 Your Input")
        input_display = input_df.rename(columns=FEATURE_LABELS).T
        input_display.columns = ["Value (cm)"]
        st.table(input_display)

    with col2:
        st.markdown("#### Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Species": [s.capitalize() for s in encoder.classes_],
            "Probability": probs,
        })
        fig = px.bar(
            prob_df, x="Species", y="Probability", color="Species",
            color_discrete_map={s.capitalize(): SPECIES_INFO[s]["color"] for s in SPECIES_INFO},
            text=prob_df["Probability"].apply(lambda p: f"{p*100:.1f}%"),
            range_y=[0, 1],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="white", height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Where your flower sits (Petal Length vs Width)")
        scatter = px.scatter(
            df, x="PetalLengthCm", y="PetalWidthCm", color="Species",
            color_discrete_map={s: SPECIES_INFO[s]["color"] for s in SPECIES_INFO},
            opacity=0.6,
        )
        scatter.add_trace(go.Scatter(
            x=[user_input["PetalLengthCm"]], y=[user_input["PetalWidthCm"]],
            mode="markers", marker=dict(size=18, color="black", symbol="star"),
            name="Your input",
        ))
        scatter.update_layout(plot_bgcolor="white", height=380)
        st.plotly_chart(scatter, use_container_width=True)

# ---- TAB 2: EXPLORE DATA ---------------------------------------------------
with tab2:
    st.markdown("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    for c, (label, val) in zip(
        [c1, c2, c3, c4],
        [("Total Samples", len(df)), ("Species", df["Species"].nunique()),
         ("Features", len(FEATURES)), ("Balanced?", "Yes ✅")]
    ):
        c.markdown(f"""<div class="metric-card"><b>{label}</b><h3 style="margin:0;">{val}</h3></div>""",
                    unsafe_allow_html=True)

    st.markdown("### Feature Distributions by Species")
    feat_pick = st.selectbox("Choose a feature", FEATURES, format_func=lambda x: FEATURE_LABELS[x])
    fig_box = px.box(
        df, x="Species", y=feat_pick, color="Species",
        color_discrete_map={s: SPECIES_INFO[s]["color"] for s in SPECIES_INFO},
    )
    fig_box.update_layout(plot_bgcolor="white", showlegend=False, height=400)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Pairwise Feature Relationships")
    fig_matrix = px.scatter_matrix(
        df, dimensions=FEATURES, color="Species",
        color_discrete_map={s: SPECIES_INFO[s]["color"] for s in SPECIES_INFO},
        labels=FEATURE_LABELS,
    )
    fig_matrix.update_layout(height=650, plot_bgcolor="white")
    fig_matrix.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=4, opacity=0.6))
    st.plotly_chart(fig_matrix, use_container_width=True)

    with st.expander("🔍 View raw data table"):
        st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit • Predictions powered by a pre-trained Keras model (model.h5)")
