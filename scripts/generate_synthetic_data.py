"""Write the default CashCast synthetic transaction dataset."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthetic_data import write_synthetic_csv


if __name__ == "__main__":
    path = write_synthetic_csv("data/synthetic/transactions.csv")
    print(f"Wrote synthetic transactions to {path}")

