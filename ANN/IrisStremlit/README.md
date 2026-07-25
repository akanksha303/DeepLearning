# 🌸 Iris Species Classifier — Streamlit App

A GUI wrapper around the modeling work in `IrisData.ipynb`: a Perceptron baseline
and a Keras Sequential Neural Network (16 → 8 → 3), trained on the classic
Iris dataset (Sepal/Petal length & width).

## What it does
- **Predict tab** — move sliders for the 4 flower measurements and get a live
  species prediction with confidence scores, from either model.
- **Explore Data tab** — box plots per feature, a full pairwise scatter matrix,
  and the raw data table.
- **Model Performance tab** — accuracy comparison, training loss/accuracy
  curves, confusion matrix, and full classification report.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Put your `Iris.csv` in the same folder as `app.py`. If it's not
   found, the app automatically falls back to the built-in scikit-learn copy
   of the Iris dataset, so it works either way — no extra setup needed.

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Your browser will open automatically at `http://localhost:8501`.

## Notes
- Models train once per session and are cached (`@st.cache_resource`), so the
  app stays fast after the first ~5–10 second load.
- If you'd rather use your own trained model files instead of retraining every
  session, say the word and I can adapt the app to load a saved `.h5`/`.pkl`
  model instead.
