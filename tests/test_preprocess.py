import pandas as pd

from src.preprocess import clean_description, prepare_transactions


def test_clean_description_removes_transfer_noise():
    assert clean_description("UPI-ZOMATO-78492") == "zomato"


def test_prepare_transactions_marks_salary_as_income():
    transactions = prepare_transactions(
        pd.DataFrame(
            {
                "Date": ["2025-01-01", "2025-01-02"],
                "Description": ["Salary Credit", "Uber Ride"],
                "Amount": [50000, 180],
            }
        )
    )

    assert transactions["FlowType"].tolist() == ["Income", "Expense"]
    assert transactions["Category"].tolist() == ["Income", ""]


def test_prepare_transactions_drops_missing_descriptions():
    transactions = prepare_transactions(
        pd.DataFrame(
            {
                "Date": ["2025-01-01", "2025-01-02"],
                "Description": [None, "Cafe Receipt"],
                "Amount": [20, 150],
            }
        )
    )

    assert transactions["Description"].tolist() == ["cafe receipt"]

