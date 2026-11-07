"""
LAB 6 - MODEL REGISTRY REPORT
"""

from pathlib import Path
import json

import mlflow
import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_NAME = (
    "LoanDefaultClassifier"
)


def generate_report():

    print("=" * 60)

    print(
        "LAB 6 - MODEL REGISTRY REPORT"
    )

    print("=" * 60)

    # Connect to MLflow
    mlflow.set_tracking_uri(

        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    client = mlflow.MlflowClient()

    # Get model versions
    versions = (
        client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )
    )

    if not versions:

        raise RuntimeError(

            f"No registered model versions "
            f"found for {MODEL_NAME}."
        )

    report = []

    for version in versions:

        aliases = (
            list(version.aliases)
            if version.aliases
            else []
        )

        report.append({

            "model_name":
                MODEL_NAME,

            "version":
                version.version,

            "run_id":
                version.run_id,

            "source":
                version.source,

            "status":
                version.status,

            "aliases":
                ", ".join(aliases),

            "created_timestamp":
                version.creation_timestamp,

            "last_updated_timestamp":
                version.last_updated_timestamp
        })

    # DataFrame
    report_df = pd.DataFrame(
        report
    )

    report_df["version"] = (
        report_df["version"]
        .astype(int)
    )

    report_df = report_df.sort_values(
        "version"
    )

    # Save CSV
    csv_path = (
        OUTPUT_DIR /
        "model_registry_report.csv"
    )

    report_df.to_csv(
        csv_path,
        index=False
    )

    # Save JSON
    json_path = (
        OUTPUT_DIR /
        "model_registry_report.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str
        )

    print("\nRegistered model versions:")

    print(
        report_df.to_string(
            index=False
        )
    )

    print(
        "\nCSV report:"
    )

    print(
        csv_path
    )

    print(
        "\nJSON report:"
    )

    print(
        json_path
    )

    print(
        "\nRegistry report generated successfully!"
    )


if __name__ == "__main__":

    generate_report()