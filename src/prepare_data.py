
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed_churn.csv"


def prepare_data():
    """Load, clean, and save the customer churn dataset."""

    # Load the raw dataset
    df = pd.read_csv(DATA_PATH)

    # Remove customer ID because it is an identifier, not a useful predictor
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges from text to numeric values
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Remove rows where TotalCharges could not be converted
    df = df.dropna(subset=["TotalCharges"])

    # Convert the target into numeric values: Yes = 1, No = 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Save the cleaned dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Data preprocessing completed successfully!")
    print(f"Processed dataset shape: {df.shape}")
    print(f"Saved to: {PROCESSED_DATA_PATH}")
    print("\nChurn distribution:")
    print(df["Churn"].value_counts().to_string())


if __name__ == "__main__":
    prepare_data()
