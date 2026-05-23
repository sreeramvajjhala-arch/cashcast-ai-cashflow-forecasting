# CashCast

CashCast is an MVP personal cash flow forecasting system. It turns bank-style
transactions into categorized spending signals, a 30-day forecast, deficit
alerts, and simple recommendations.

## Architecture

```text
CSV / synthetic transactions
        |
        v
Cleaning and transaction normalization
        |
        v
TF-IDF + Logistic Regression categorizer
        |
        v
Daily category spending features
        |
        v
Prophet forecast
        |
        v
Risk and recommendation engines
        |
        v
Streamlit dashboard
```

The first version favors interpretable components over model complexity. The
classifier can be evaluated with standard NLP metrics and the forecast can be
compared against held-out daily spending with MAE, RMSE, and MAPE.

## Quick start

Create a virtual environment with a working Python 3.11 or 3.12 interpreter,
then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Generate a year of synthetic transactions:

```powershell
python scripts\generate_synthetic_data.py
```

Train and save the baseline expense categorizer:

```powershell
python scripts\train_classifier.py
```

Run the dashboard:

```powershell
streamlit run app\streamlit_app.py
```

The app loads `data/synthetic/transactions.csv` when it exists. It also accepts
a CSV with `Date`, `Description`, and `Amount` columns. An optional `Category`
column is useful for training and evaluation. To keep the demo responsive, the
dashboard defaults to a fast moving-average baseline over the four leading spend
categories. Prophet remains available from the sidebar for the richer ML
forecasting path.

## Data contract

`src.preprocess.prepare_transactions` normalizes uploaded CSVs into these
columns:

| Column | Meaning |
| --- | --- |
| Date | Parsed transaction date |
| Description | Cleaned transaction text |
| Amount | Absolute transaction amount |
| FlowType | `Expense` or `Income` |
| Category | Uploaded, inferred, or predicted category |

Amounts are normalized to positive values. A row is treated as income when its
description or supplied category looks like income; all other rows remain
expenses until the statement adapter grows a richer debit/credit contract.

## Repository layout

```text
app/                    Streamlit UI
data/                   Raw, processed, and generated data folders
models/                 Saved model artifacts
scripts/                Dataset generation entry points
src/                    Cleaning, modeling, risk, and insight code
tests/                  Focused unit tests
```

The baseline classifier is saved as `models/classifier.joblib`. Forecast
experiments can use `src.evaluate.forecast_metrics` to record MAE, RMSE, and
MAPE after aligning held-out actual and predicted daily spending.

## Suggested experiments

1. Hold out the latest month before comparing Prophet forecasts.
2. Compare categorization rules against the TF-IDF pipeline.
3. Add behavioral features for recurring bills, impulse spending, and lifestyle
   inflation once the baseline metrics are recorded.
