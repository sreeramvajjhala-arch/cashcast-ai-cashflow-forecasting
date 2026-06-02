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

## Dataset

For the MVP, CashCast uses synthetic transaction data.

The generator includes salary cycles, rent, subscriptions, weekend spending, utilities, shopping, and seasonal spending spikes.

## Transaction Preprocessing

Bank statement descriptions are often noisy and inconsistent.

Examples:

- UPI-ZOMATO-78492
- SWIGGYPAY-1122
- AMAZONINDIA-PAY-8832
- NEFT-HDFC-RENT

CashCast cleans transaction descriptions by removing transfer noise, transaction IDs, special characters, and inconsistent formatting.

## Expense Categorization

For the MVP, CashCast uses a TF-IDF + Logistic Regression pipeline for expense categorization.

This approach was chosen because it is:

- fast
- explainable
- suitable for small datasets
- easy to deploy inside a Streamlit app

This allows the forecasting and categorization pipeline to be tested without using sensitive personal bank data.
## Forecasting Approach

CashCast forecasts category-wise spending instead of raw transactions.

This makes the output more actionable.

Example:

- Food: ₹7,800
- Transport: ₹2,400
- Shopping: ₹3,200
- Utilities: ₹2,100

The MVP includes two forecasting paths:

1. Fast baseline forecast  
   Uses recent daily category averages. This keeps the dashboard responsive.

2. Prophet forecast  
   Models trend and seasonality for richer time-series forecasting.

## Evaluation Metrics

Forecasting experiments can be evaluated using:

- MAE
- RMSE
- MAPE
