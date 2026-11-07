"""
LAB 5 - OUTPUT VALIDATION
Validates the processed ML datasets.
"""

from pathlib import Path
import json
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def validate_outputs():

    print("=" * 60)
    print("LAB 5 - OUTPUT VALIDATION")
    print("=" * 60)

    required_files = [

        "X_train_final.npy",

        "X_test_final.npy",

        "y_train.npy",

        "y_test.npy"
    ]

    # Check files
    for filename in required_files:

        file_path = (
            PROCESSED_DIR / filename
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

    # Load data
    X_train = np.load(
        PROCESSED_DIR / "X_train_final.npy"
    )

    X_test = np.load(
        PROCESSED_DIR / "X_test_final.npy"
    )

    y_train = np.load(
        PROCESSED_DIR / "y_train.npy"
    )

    y_test = np.load(
        PROCESSED_DIR / "y_test.npy"
    )

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # Validation checks
    checks = {

        "train_is_2D":
            X_train.ndim == 2,

        "test_is_2D":
            X_test.ndim == 2,

        "same_number_of_features":
            X_train.shape[1] == X_test.shape[1],

        "train_rows_match_target":
            X_train.shape[0] == len(y_train),

        "test_rows_match_target":
            X_test.shape[0] == len(y_test),

        "train_has_no_nan":
            not np.isnan(X_train).any(),

        "test_has_no_nan":
            not np.isnan(X_test).any(),

        "train_has_no_infinity":
            not np.isinf(X_train).any(),

        "test_has_no_infinity":
            not np.isinf(X_test).any()
    }

    validation_passed = all(
        checks.values()
    )

    report = {

        "train_shape":
            list(X_train.shape),

        "test_shape":
            list(X_test.shape),

        "y_train_shape":
            list(y_train.shape),

        "y_test_shape":
            list(y_test.shape),

        "checks":
            checks,

        "validation_passed":
            validation_passed
    }

    report_path = (
        OUTPUT_DIR /
        "output_validation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\nValidation checks:")

    for check, result in checks.items():

        status = "PASS" if result else "FAIL"

        print(f"{check}: {status}")

    if not validation_passed:

        raise ValueError(
            "Output validation failed."
        )

    print(
        "\nOutput validation "
        "completed successfully!"
    )


if __name__ == "__main__":
    validate_outputs()