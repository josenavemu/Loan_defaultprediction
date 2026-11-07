"""
LAB 6 - MODEL LIFECYCLE AUTOMATION
"""

from pathlib import Path

import mlflow


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

MODEL_NAME = (
    "LoanDefaultClassifier"
)


def automate_lifecycle():

    print("=" * 60)

    print(
        "LAB 6 - MODEL LIFECYCLE AUTOMATION"
    )

    print("=" * 60)

    # Connect to MLflow
    mlflow.set_tracking_uri(

        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    client = mlflow.MlflowClient()

    # Get registered model versions
    versions = (
        client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )
    )

    if not versions:

        raise RuntimeError(

            f"No registered versions found "
            f"for {MODEL_NAME}.\n"
            "Run train_registry.py first."
        )

    # Find latest version
    latest_version = max(

        versions,

        key=lambda x:
            int(x.version)
    )

    version = latest_version.version

    print(
        f"Latest model version: {version}"
    )

    # Staging alias
    client.set_registered_model_alias(

        MODEL_NAME,

        "staging",

        version
    )

    print(
        f"Version {version} "
        "assigned to STAGING."
    )

    # Production alias
    client.set_registered_model_alias(

        MODEL_NAME,

        "production",

        version
    )

    print(
        f"Version {version} "
        "assigned to PRODUCTION."
    )

    print("\n" + "=" * 60)

    print(
        "MODEL LIFECYCLE COMPLETED"
    )

    print("=" * 60)

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Staging : {MODEL_NAME}@staging"
    )

    print(
        f"Production: {MODEL_NAME}@production"
    )


if __name__ == "__main__":

    automate_lifecycle()