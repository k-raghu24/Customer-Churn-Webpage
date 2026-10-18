# Customer Churn Prediction Website

A Flask dashboard for your Telco Customer Churn project. It uses a Random Forest classifier inside a scikit-learn preprocessing pipeline.

## 1. Put the dataset in the project folder

Copy `Telco_customer_churn.xlsx` into the same folder as `app.py` and `train_model.py`.

The notebook's source dataset has 7,043 rows. Your uploaded notebook does not include the Excel dataset itself, so this project does not bundle it.

## 2. Create and activate a virtual environment (Windows)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

## 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 4. Train and save the model

```bash
python train_model.py
```

This creates:
- `models/churn_pipeline.joblib`
- `models/metrics.json`

The preprocessing is fitted only on the training split and saved together with the classifier. Do not manually encode form values or scale the input again; the pipeline handles that.

## 5. Start the website

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Feature and leakage decisions

The target is `Churn Value`. The model excludes:
- `Churn Value` from inputs (the target)
- `Churn Label`, `Churn Score`, and `Churn Reason` because they describe or are closely tied to the churn outcome
- customer IDs and geographic columns that do not make useful general prediction inputs
- `CLTV`, because this version keeps the form to attributes that can be entered independently of the outcome; verify its business definition before including it.

`Total Charges` is converted to numeric. Missing values are imputed inside the pipeline; missing total charges are first filled using monthly charges × tenure, consistent with the notebook's treatment.

## Evaluation

The training script reports held-out test accuracy, churn precision, churn recall, churn F1, ROC-AUC and the confusion matrix. The uploaded notebook's original split reported:
- Logistic Regression accuracy: 80.27%
- KNN accuracy: 76.65%
- Random Forest accuracy: 79.06%
- Random Forest churn recall: 51%

The new pipeline uses stratified splitting, one-hot encoding and class weighting, so its results will differ. Compare churn recall/F1 as well as accuracy; accuracy alone can hide missed churners.

## Notes

- This is a local demo. `debug=True` is suitable only for local development; turn it off before public deployment.
- The probability is a model estimate, not a guarantee or a calibrated business probability.
- The HTML options are designed for the standard Telco Customer Churn dataset. If your Excel file has different category spellings, update `OPTIONS` in `app.py`.
