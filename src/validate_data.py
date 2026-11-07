"""
LAB 5 - DATA VALIDATION
Validates the raw Loan Default dataset before preprocessing.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = Path(
    r"C:\Users\user\Downloads\Loan_Default.csv (1)\Loan_Default.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def validate_data():

    print("=" * 60)
    print("LAB 5 - DATA VALIDATION")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # Basic information
    duplicate_rows = int(df.duplicated().sum())
    missing_values = int(df.isna().sum().sum())

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # Validation checks
    checks = {
        "dataset_not_empty": len(df) > 0,
        "target_column_exists": "Status" in df.columns,
        "no_all_null_columns": not any(df.isna().all()),
        "no_duplicate_column_names":
            len(df.columns) == len(set(df.columns))
    }

    validation_passed = all(checks.values())

    # Target distribution
    if "Status" in df.columns:
        target_distribution = (
            df["Status"]
            .value_counts(dropna=False)
            .to_dict()
        )
    else:
        target_distribution = {}

    # Create report
    report = {
        "dataset": "Loan Default",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "missing_values_total": missing_values,
        "target_column": "Status",
        "target_exists": "Status" in df.columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_distribution": target_distribution,
        "checks": checks,
        "validation_passed": validation_passed
    }

    # Save report
    report_path = (
        PROCESSED_DIR / "validation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            default=str
        )

    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Missing values: {missing_values}")

    print("\nValidation checks:")

    for check, result in checks.items():

        status = "PASS" if result else "FAIL"

        print(f"{check}: {status}")

    if not validation_passed:

        raise ValueError(
            "Data validation failed. "
            "Check validation_report.json"
        )

    print("\nData validation completed successfully!")


if __name__ == "__main__":
    validate_data()