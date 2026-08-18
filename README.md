# ML Assignment 2 — Classification Models + Streamlit Deployment

End-to-end machine-learning workflow: train five classification models on a public
dataset, evaluate them with six metrics, and serve the results through an
interactive Streamlit app deployed on Streamlit Community Cloud.

---

## a. Problem statement

Given 30 numeric measurements computed from a digitised image of a fine-needle
aspirate (FNA) of a breast mass, predict whether the tumour is **malignant** or
**benign**. This is a **binary classification** problem. Reliable automatic
screening supports early cancer detection, where correctly flagging malignant
cases (high recall) is especially important. We train five classifiers, compare
them on six metrics, and expose them in a web app where a user can upload test
data, pick a model, and view its metrics and confusion matrix.

## b. Dataset description

- **Name:** Breast Cancer Wisconsin (Diagnostic) — a public **UCI** dataset
  (also bundled with scikit-learn as `load_breast_cancer`).
- **Instances:** 569 &nbsp;(≥ 500 ✓)
- **Features:** 30 real-valued features &nbsp;(≥ 12 ✓) — the mean, standard error
  and "worst" of ten nuclear measurements (radius, texture, perimeter, area,
  smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
- **Target:** binary. The scikit-learn target is `0 = malignant, 1 = benign`; we
  **relabel** it so the **positive class (1) = malignant (cancer)**, which makes
  precision/recall directly meaningful for cancer screening.
- **Class balance:** 212 malignant (37.3%) / 357 benign (62.7%).
- **Split:** stratified 80/20 → 455 train / 114 test (`random_state = 42`).
  `train_data.csv` and `test_data.csv` are included in the repo; the app uploads
  **test data only** (per the free-tier guidance).

## c. GitHub repository link

`https://github.com/aniruddh-dse/breast-cancer-prediction.git`

Repository contents:

```
project-folder/
├── app.py                 # Streamlit application
├── requirements.txt       # dependencies for deployment
├── README.md              # this file
├── test_data.csv          # held-out test set (upload this in the app)
├── train_data.csv         # training set (used by the training script / app fallback)
└── model/
    ├── train_models.py     # training "notebook" (.py) — trains, evaluates, saves everything
    ├── metrics.json        # saved metrics used in the table below
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

## d. Models used

Five classifiers were trained on the **same** dataset. Logistic Regression and kNN
run inside a `StandardScaler` pipeline (they are scale-sensitive); Decision Tree,
Random Forest and Gaussian Naive Bayes are scale-invariant and used directly.

### Comparison table (evaluated on the 114-row test set)

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|:--------:|:------:|:---------:|:------:|:------:|:------:|
| Logistic Regression    | 0.9649   | 0.9960 | 0.9750    | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree          | 0.9298   | 0.9246 | 0.9048    | 0.9048 | 0.9048 | 0.8492 |
| kNN                    | 0.9561   | 0.9823 | 0.9744    | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes            | 0.9386   | 0.9934 | 1.0000    | 0.8333 | 0.9091 | 0.8715 |
| Random Forest (Ensemble) | 0.9649 | 0.9942 | 1.0000    | 0.9048 | 0.9500 | 0.9258 |

*(Precision / Recall / F1 are for the positive class = malignant. Metrics are also
stored in `model/metrics.json` and reproduced live in the Streamlit app.)*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Best overall: highest **AUC (0.996)** and highest **F1 (0.951)** with well-balanced precision/recall. After standardisation the two classes are almost linearly separable, so a simple linear boundary generalises very well at minimal cost. |
| **Decision Tree** | Weakest of the five (F1 0.905, MCC 0.849, AUC 0.925). A single unpruned tree overfits and produces coarse, step-like probability estimates, which drags its AUC down and increases variance. |
| **kNN** | Strong once features are scaled (F1 0.938, AUC 0.982). Being distance-based, it depends critically on the `StandardScaler` step; recall (0.905) is slightly lower as it misses a few malignant cases near class boundaries. |
| **Naive Bayes** | **Perfect precision (1.000)** but the **lowest recall (0.833)** — its feature-independence assumption (features here are correlated) makes it conservative, so it misses the most malignant cases. High AUC (0.993) shows its probability ranking is still good. |
| **Random Forest (Ensemble)** | Essentially tied for best: **highest MCC (0.926)**, perfect precision, F1 0.950, AUC 0.994. Averaging many trees collapses the single tree's variance (MCC 0.849 → 0.926), giving a robust, high-precision model. |
| **Overall Winner for your dataset?** | **Logistic Regression** — top F1 (0.9512) and AUC (0.9960), with **Random Forest a very close second** (marginally higher MCC). For this dataset the simplest model wins: it matches the ensemble's accuracy at far lower complexity. If maximising recall (catching every cancer) were the sole objective, Logistic Regression also had the best recall (0.929) among the leaders, though none achieved perfect recall. |

## Streamlit app features

The deployed app (`app.py`) implements every required feature:

- **CSV upload** for test data (defaults to the bundled `test_data.csv` so results
  are visible immediately).
- **Model-selection dropdown** for the five models.
- **Display of all six metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) for the
  selected model.
- **Confusion matrix** (heatmap) **and classification report**.
- **Bonus:** an all-models comparison table computed live on the current test data,
  highlighting the best value per metric.

It loads the saved `model/*.pkl` files and **automatically retrains from
`train_data.csv`** if the pickles cannot be loaded (e.g. a library-version mismatch
on the cloud), so deployment does not fail.

## Live Streamlit app link

`https://fsza3z2lu24fxr6bbfc2su.streamlit.app/'

## How to run

**Locally**
```bash
pip install -r requirements.txt
python model/train_models.py      # (re)generate CSVs, metrics.json and the .pkl models
streamlit run app.py
```

**Deploy (Streamlit Community Cloud)**
1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud → sign in with GitHub → **New App**.
3. Select the repo, branch `main`, and `app.py`. Click **Deploy**.

## Reproducibility & academic integrity

All results use `random_state = 42` and are fully reproducible via
`python model/train_models.py`. The code, pipeline design, UI and observations are
original work written specifically for this assignment.
