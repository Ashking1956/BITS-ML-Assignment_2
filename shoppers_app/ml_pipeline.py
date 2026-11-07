"""ml_pipeline.py - the whole model pipeline in ONE file.

Train once (saves everything the Streamlit app needs):
    python ml_pipeline.py                    # uses online_shoppers_intention.csv next to this file
    python ml_pipeline.py path/to/file.csv

In the app:
    import ml_pipeline as mp                 # importing does NOT train anything

Writes:
    model/<name>.joblib         one fitted model (scaler + classifier) per model
    model/feature_columns.json  column order the models expect
    model/raw_columns.json      raw input columns an uploaded test CSV must contain
    model/metrics.csv           test-set metrics of the 5 models
    test_data.csv               the hold-out rows, in the original column format
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
DATA_FILE = "online_shoppers_intention.csv"
TARGET = "Revenue"
RANDOM_STATE = 42
TEST_SIZE = 0.2

METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"]


# ------------------------------------------------------------------- columns
def detect_columns(df):
    """Find the column groups from the data itself. Returns (features, num_cols, cat_cols)."""
    features = [c for c in df.columns if c != TARGET]
    cat_cols, num_cols = [], []
    for c in features:
        s = df[c]
        if s.dtype == bool or not pd.api.types.is_numeric_dtype(s):
            cat_cols.append(c)       # text or True/False
        elif pd.api.types.is_integer_dtype(s) and s.min() >= 1 and s.nunique() == s.max():
            cat_cols.append(c)       # ID codes 1..k with no gaps (Browser, Region ...)
        else:
            num_cols.append(c)       # real numbers and counts
    return features, num_cols, cat_cols

def text_columns(df):
    """Columns that hold text (these get one-hot encoded)."""
    return [c for c in df.columns if df[c].dtype != bool and not pd.api.types.is_numeric_dtype(df[c])]

def to_int_bools(df):
    """True/False columns -> 1/0."""
    return df.astype({c: int for c in df.select_dtypes("bool").columns})

# ------------------------------------------------------------------- data steps
def clean(raw):
    """Remove duplicates and turn True/False columns into 1/0."""
    return to_int_bools(raw.drop_duplicates().reset_index(drop=True))

def engineer(df):
    """Add the 7 engineered features."""
    fe = df.copy()
    fe["TotalPages"] = fe["Administrative"] + fe["Informational"] + fe["ProductRelated"]
    fe["TotalDuration"] = (fe["Administrative_Duration"] + fe["Informational_Duration"] + fe["ProductRelated_Duration"])
    fe["AvgTimePerPage"] = (fe["TotalDuration"] / fe["TotalPages"].replace(0, np.nan)).fillna(0)
    fe["HasPageValue"] = (fe["PageValues"] > 0).astype(int)
    fe["IsReturning"] = (fe["VisitorType"] == "Returning_Visitor").astype(int)
    fe["IsHolidaySeason"] = fe["Month"].isin(["Nov", "Dec"]).astype(int)
    fe["log_ProductRelated_Duration"] = np.log1p(fe["ProductRelated_Duration"])
    return fe

def encode(fe):
    """One-hot encode the text columns (drop_first=True), True/False -> 1/0."""
    return to_int_bools(pd.get_dummies(fe, columns=text_columns(fe), drop_first=True))

def split(enc):
    """Stratified train/test split."""
    X, y = enc.drop(columns=[TARGET]), enc[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

def prepare_for_saved_models(df, feature_columns):
    """Turn an uploaded test CSV into the feature matrix the saved models expect.

    Accepts the raw dataset columns, or already-processed columns. No duplicate removal and no
    re-fitting: only the same row-wise steps as training.
    Returns (X, y); y is None if the target column is missing.
    """
    df = df.copy()
    y = df[TARGET].astype(int) if TARGET in df.columns else None
    if all(c in df.columns for c in feature_columns): # already processed
        return df[feature_columns], y

    raw_columns = json.loads((MODEL_DIR / "raw_columns.json").read_text())
    missing = [c for c in raw_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    fe = engineer(to_int_bools(df))
    enc = to_int_bools(pd.get_dummies(fe, columns=text_columns(fe)))
    # the category dropped at training time has no column -> it is simply all zeros
    return enc.reindex(columns=feature_columns, fill_value=0), y

# ------------------------------------------------------------------- models
def build_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "KNN":                 make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes":         make_pipeline(StandardScaler(), GaussianNB()),
        "SVM":                 make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_STATE)),
    }

def continuous_score(model, X):
    """Probability if the model has one, otherwise the decision function (SVM)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

def evaluate(model, X_test, y_test):
    """Predict on data the model was not trained on and compute the 6 metrics."""
    pred = model.predict(X_test)
    score = continuous_score(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, score)
    two_classes = len(set(y_test)) == 2
    return {
        "metrics": {
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, score) if two_classes else float("nan"),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        },
        "cm": confusion_matrix(y_test, pred, labels=[0, 1]),
        "fpr": fpr, "tpr": tpr, "pred": pred, "score": score,
    }

def fit_and_evaluate(name, X_train, y_train, X_test, y_test):
    model = build_models()[name]
    model.fit(X_train, y_train)
    return model, evaluate(model, X_test, y_test)

# ------------------------------------------------------------------- training script
def slug(name):
    return name.lower().replace(" ", "_")

def train(csv_path=ROOT / DATA_FILE):
    raw = pd.read_csv(csv_path)
    features, num_cols, cat_cols = detect_columns(raw)
    df = clean(raw)
    fe = engineer(df)

    X_train, X_test, y_train, y_test = split(encode(fe))
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

    MODEL_DIR.mkdir(exist_ok=True)
    rows = []
    for name in MODEL_NAMES:
        model, res = fit_and_evaluate(name, X_train, y_train, X_test, y_test)
        joblib.dump(model, MODEL_DIR / f"{slug(name)}.joblib")
        rows.append({"Model": name, **res["metrics"]})
        print(f"saved {slug(name)}.joblib")

    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(list(X_train.columns)))
    (MODEL_DIR / "raw_columns.json").write_text(json.dumps(features))
    results = pd.DataFrame(rows).set_index("Model").round(4)
    results.to_csv(MODEL_DIR / "metrics.csv")

    # hold-out rows in the original raw format (same rows as the split)
    df.loc[X_test.index].to_csv(ROOT / "test_data.csv", index=False)
    print(f"test_data.csv: {len(X_test)} rows\n")
    print(results.to_string())

if __name__ == "__main__":
    train(sys.argv[1] if len(sys.argv) > 1 else ROOT / DATA_FILE)
