from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Telco_customer_churn.xlsx"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "Churn Value"

# These columns are excluded because they are identifiers, geographic fields,
# post-outcome fields, or direct/near-direct descriptions of the churn outcome.
DROP_COLUMNS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude", "Gender",
    "Churn Label", "Churn Score", "Churn Reason"
]

# CLTV is excluded here to keep the form limited to customer/service details
# available independently of the churn outcome. It can be reconsidered only
# if its definition and availability at prediction time are verified.
EXCLUDE_FROM_FEATURES = DROP_COLUMNS + [TARGET, "CLTV"]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Copy Telco_customer_churn.xlsx into the project folder."
        )

    df = pd.read_excel(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in the dataset.")

    # Convert charges to numeric; the source dataset may contain blank strings.
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    if "Monthly Charges" in df.columns and "Tenure Months" in df.columns:
        fallback = df["Monthly Charges"] * df["Tenure Months"]
        df["Total Charges"] = df["Total Charges"].fillna(fallback)

    # Target must be binary 0/1. Do not use target-related columns as inputs.
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    X = df.drop(columns=[c for c in EXCLUDE_FROM_FEATURES if c in df.columns])

    if TARGET in X.columns:
        X = X.drop(columns=[TARGET])

    # Make sure no outcome-related fields slipped into the model.
    forbidden = {"Churn Value", "Churn Label", "Churn Score", "Churn Reason"}
    assert forbidden.isdisjoint(X.columns), "Target leakage column detected."

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Random Forest",
        "test_rows": int(len(y_test)),
        "feature_count": int(X.shape[1]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_churn": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_churn": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_churn": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Stay", "Churn"],
            output_dict=True, zero_division=0
        ),
        "features": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET,
        "note": "Metrics are from a held-out 20% stratified test split."
    }

    joblib.dump({
        "pipeline": pipeline,
        "features": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET
    }, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel saved to:", MODEL_PATH)
    print("\nHeld-out test metrics")
    print(f"Accuracy:       {metrics['accuracy']:.3f}")
    print(f"Churn precision:{metrics['precision_churn']:.3f}")
    print(f"Churn recall:   {metrics['recall_churn']:.3f}")
    print(f"Churn F1:       {metrics['f1_churn']:.3f}")
    print(f"ROC-AUC:        {metrics['roc_auc']:.3f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print("\nImportant: compare churn recall and F1, not accuracy alone.")

if __name__ == "__main__":
    main()
