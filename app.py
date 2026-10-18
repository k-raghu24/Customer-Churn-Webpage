from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run `python train_model.py` first."
    )

saved = joblib.load(MODEL_PATH)
pipeline = saved["pipeline"]
features = saved["features"]
numeric_features = set(saved["numeric_features"])
categorical_features = set(saved["categorical_features"])

if METRICS_PATH.exists():
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
else:
    metrics = {}

# These choices match the standard Telco customer churn dataset.
OPTIONS = {
    "Senior Citizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "Phone Service": ["No", "Yes"],
    "Multiple Lines": ["No", "Yes", "No phone service"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": ["No", "Yes", "No internet service"],
    "Online Backup": ["No", "Yes", "No internet service"],
    "Device Protection": ["No", "Yes", "No internet service"],
    "Tech Support": ["No", "Yes", "No internet service"],
    "Streaming TV": ["No", "Yes", "No internet service"],
    "Streaming Movies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Paperless Billing": ["No", "Yes"],
    "Payment Method": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
}

DEFAULTS = {
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": "12",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": "70.00",
    "Total Charges": "840.00"
}

NUMERIC_LIMITS = {
    "Tenure Months": (0, 72),
    "Monthly Charges": (0, 200),
    "Total Charges": (0, 100000)
}

def parse_form(form):
    row = {}
    errors = []

    for feature in features:
        raw = form.get(feature, "").strip()

        if feature in numeric_features:
            if raw == "":
                errors.append(f"{feature} is required.")
                continue
            try:
                value = float(raw)
            except ValueError:
                errors.append(f"{feature} must be a number.")
                continue

            if feature in NUMERIC_LIMITS:
                low, high = NUMERIC_LIMITS[feature]
                if value < low or value > high:
                    errors.append(f"{feature} must be between {low} and {high}.")
                    continue
            row[feature] = value
        else:
            if raw not in OPTIONS.get(feature, []):
                errors.append(f"Choose a valid option for {feature}.")
                continue
            row[feature] = raw

    return row, errors

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error_message = None
    form_values = DEFAULTS.copy()

    if request.method == "POST":
        form_values.update({k: v for k, v in request.form.items()})
        row, errors = parse_form(request.form)

        if errors:
            error_message = " ".join(errors)
        else:
            # Explicit feature order matches the trained pipeline.
            input_df = pd.DataFrame([row], columns=features)
            prediction = int(pipeline.predict(input_df)[0])
            probability = float(pipeline.predict_proba(input_df)[0][1])
            result = {
                "prediction": prediction,
                "label": "Likely to churn" if prediction == 1 else "Likely to stay",
                "probability": round(probability * 100, 1),
                "risk": (
                    "High" if probability >= 0.65
                    else "Medium" if probability >= 0.35
                    else "Low"
                ),
                "message": (
                    "Consider proactive retention support and a review of the customer's service experience."
                    if prediction == 1 else
                    "The model predicts this customer is more likely to remain. Continue normal customer care."
                )
            }

    return render_template(
        "index.html",
        options=OPTIONS,
        defaults=form_values,
        result=result,
        error_message=error_message,
        metrics=metrics
    )

if __name__ == "__main__":
    app.run(debug=True)
