"""
ML Assignment 2 - Streamlit app
================================
Interactive front-end for 5 classification models trained on the Breast Cancer
Wisconsin (Diagnostic) dataset (positive class = malignant).

Features (as required by the assignment):
  a. CSV upload for TEST data (falls back to the bundled test_data.csv)
  b. Model-selection dropdown
  c. Display of all 6 evaluation metrics
  d. Confusion matrix + classification report
Plus a bonus all-models comparison table on the current test data.

Robustness: the app first tries to load the pre-trained model/*.pkl files. If that
fails for any reason (e.g. a scikit-learn version mismatch on the cloud), it
transparently retrains the same pipelines from train_data.csv (cached), so the app
never crashes on deployment.
"""

import io
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import joblib

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COL = "target"
CLASS_LABELS = ["benign (0)", "malignant (1)"]
RANDOM_STATE = 42

MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
}

st.set_page_config(page_title="Breast Cancer Classifier Comparison",
                   page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------
# Model loading / training (cached so it runs once per session)
# --------------------------------------------------------------------------
def _build_pipelines():
    """Fresh, unfitted versions of the 5 required models (same as training)."""
    return {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=5000,
                                                                    random_state=RANDOM_STATE))]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": Pipeline([("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200,
                                                 random_state=RANDOM_STATE),
    }


@st.cache_resource(show_spinner=False)
def load_models():
    """
    Return (models_dict, source_str).
    Try the saved .pkl files first; if any is missing or fails to load, retrain
    all pipelines from train_data.csv so the app always has working models.
    """
    # 1) attempt to load every saved model
    try:
        models = {}
        for name, key in MODEL_FILES.items():
            models[name] = joblib.load(MODEL_DIR / f"{key}.pkl")
        return models, "loaded pre-trained model/*.pkl"
    except Exception as exc:  # noqa: BLE001  (any failure -> retrain fallback)
        fallback_reason = str(exc)

    # 2) fallback: retrain from the bundled training CSV
    train_path = ROOT / "train_data.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            "No saved models and no train_data.csv to retrain from. "
            f"Original load error: {fallback_reason}")
    tdf = pd.read_csv(train_path)
    X_tr, y_tr = tdf.drop(columns=[TARGET_COL]), tdf[TARGET_COL]
    models = _build_pipelines()
    for m in models.values():
        m.fit(X_tr, y_tr)
    return models, "retrained from train_data.csv (saved models unavailable)"


@st.cache_data(show_spinner=False)
def load_default_test():
    """Load the bundled test_data.csv, if present."""
    p = ROOT / "test_data.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_feature_order():
    """Feature column order the models expect (from train_data.csv header)."""
    p = ROOT / "train_data.csv"
    if p.exists():
        cols = list(pd.read_csv(p, nrows=1).columns)
        return [c for c in cols if c != TARGET_COL]
    return None


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def compute_metrics(model, X, y):
    """All six required metrics for one fitted model (AUC guarded for 1-class sets)."""
    y_pred = model.predict(X)
    try:
        y_score = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_score)
    except Exception:  # e.g. only one class present in an uploaded subset
        auc = float("nan")
    return {
        "Accuracy":  accuracy_score(y, y_pred),
        "AUC":       auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall":    recall_score(y, y_pred, zero_division=0),
        "F1":        f1_score(y, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y, y_pred),
    }


def split_features_target(df, feature_order):
    """Return (X, y, message). y is None if no target column is available."""
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
        msg = ""
    else:
        # assume the last column is the label if 'target' is missing
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        msg = (f"No '{TARGET_COL}' column found - assuming the last column "
               f"('{df.columns[-1]}') holds the true labels.")
    # align feature columns to the training order when possible
    if feature_order is not None and set(feature_order).issubset(set(X.columns)):
        X = X[feature_order]
    return X, y, msg


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🩺 Breast Cancer Classifier — Model Comparison")
st.markdown(
    "Compare five classification models on the **Breast Cancer Wisconsin "
    "(Diagnostic)** dataset. Upload a test CSV (or use the bundled one), pick a "
    "model, and inspect its metrics, confusion matrix and classification report. "
    "The positive class is **malignant (1)**."
)

models, model_source = load_models()
feature_order = load_feature_order()

with st.sidebar:
    st.header("⚙️ Controls")
    up = st.file_uploader("Upload TEST data (CSV, must include a 'target' column)",
                          type=["csv"])
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    st.caption(f"Model source: {model_source}")
    st.caption("Tip: use the repo's test_data.csv for the intended results.")

# resolve which dataframe to use
if up is not None:
    data = pd.read_csv(up)
    data_source = "uploaded file"
else:
    data = load_default_test()
    data_source = "bundled test_data.csv"

if data is None:
    st.warning("No data available. Upload a test CSV to continue.")
    st.stop()

X, y, note = split_features_target(data, feature_order)
if note:
    st.info(note)

st.write(f"**Data source:** {data_source} &nbsp;|&nbsp; **rows:** {len(data)} "
         f"&nbsp;|&nbsp; **features:** {X.shape[1]}")

# ---- selected-model results ------------------------------------------------
st.subheader(f"📊 Results — {model_name}")
model = models[model_name]

try:
    m = compute_metrics(model, X, y)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not score this data with the model: {exc}")
    st.stop()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy",  f"{m['Accuracy']:.3f}")
c2.metric("AUC",       "N/A" if np.isnan(m["AUC"]) else f"{m['AUC']:.3f}")
c3.metric("Precision", f"{m['Precision']:.3f}")
c4.metric("Recall",    f"{m['Recall']:.3f}")
c5.metric("F1",        f"{m['F1']:.3f}")
c6.metric("MCC",       f"{m['MCC']:.3f}")

# ---- confusion matrix + classification report ------------------------------
left, right = st.columns([1, 1])
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, model.predict(X))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r", cbar=False,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y, model.predict(X),
                                target_names=CLASS_LABELS, zero_division=0,
                                output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# ---- all-models comparison on the current data -----------------------------
st.subheader("🏁 All models on this test data")
rows = {name: compute_metrics(mdl, X, y) for name, mdl in models.items()}
comp = pd.DataFrame(rows).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
comp = comp.round(4)
st.dataframe(
    comp.style.highlight_max(axis=0, color="#1b5e20").format("{:.4f}"),
    use_container_width=True,
)
best = comp["F1"].idxmax()
st.success(f"Best model on this data by F1-score: **{best}** "
           f"(F1 = {comp.loc[best, 'F1']:.4f}, MCC = {comp.loc[best, 'MCC']:.4f}).")

with st.expander("About this app / dataset"):
    st.markdown(
        "- **Dataset:** Breast Cancer Wisconsin (Diagnostic), 569 instances, 30 "
        "features, binary. Positive class relabelled to malignant (1).\n"
        "- **Models:** Logistic Regression, Decision Tree, kNN, Gaussian Naive "
        "Bayes, Random Forest.\n"
        "- **Metrics:** Accuracy, AUC, Precision, Recall, F1, MCC.\n"
        "- Logistic Regression / kNN run inside a StandardScaler pipeline; the "
        "saved models therefore accept raw feature CSVs directly."
    )
