"""
LAB 5 - REPRODUCIBLE PREPROCESSING PIPELINE
"""

from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREPROCESS_SCRIPT = (
    PROJECT_ROOT / "src" / "preprocess.py"
)

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def run_preprocessing_pipeline():

    print("=" * 60)
    print("LAB 5 - REPRODUCIBLE PREPROCESSING PIPELINE")
    print("=" * 60)

    # Configuration used by the project
    config = {

        "dataset": "Loan Default",

        "target": "Status",

        "test_size": 0.20,

        "random_state": 42,

        "imputation": "Median",

        "scaling": "StandardScaler",

        "categorical_encoding":
            "OneHotEncoder",

        "class_balancing":
            "SMOTE",

        "correlation_threshold": 0.85,

        "fit_preprocessors_on_training_data_only":
            True
    }

    # Save configuration
    config_path = (
        PROCESSED_DIR /
        "preprocessing_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    print("Running preprocess.py...")

    result = subprocess.run(

        [
            sys.executable,
            str(PREPROCESS_SCRIPT)
        ],

        cwd=PROJECT_ROOT,

        check=False
    )

    if result.returncode != 0:

        raise RuntimeError(
            "preprocess.py failed."
        )

    # Files expected after preprocessing
    required_files = [

        "X_train_final.npy",

        "X_test_final.npy",

        "y_train.npy",

        "y_test.npy",

        "dataset_metadata.json",

        "feature_names.json",

        "preprocessing_config.json"
    ]

    missing_files = []

    for filename in required_files:

        file_path = (
            PROCESSED_DIR / filename
        )

        if not file_path.exists():

            missing_files.append(filename)

    if missing_files:

        print("\nMissing files:")

        for file in missing_files:
            print(file)

        raise FileNotFoundError(
            "Some preprocessing output files are missing."
        )

    print("\nAll preprocessing files verified.")

    print(
        "\nReproducible preprocessing "
        "pipeline completed successfully!"
    )


if __name__ == "__main__":
    run_preprocessing_pipeline()