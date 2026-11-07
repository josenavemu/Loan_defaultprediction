from pathlib import Path
import mlflow

project = Path.cwd()
database = project / "mlflow.db"

mlflow.set_tracking_uri("sqlite:///" + database.as_posix())

client = mlflow.MlflowClient()

versions = client.search_model_versions(
    "name = 'LoanDefaultClassifier'"
)

for version in versions:
    print(
        "Version:", version.version,
        "| Aliases:", list(version.aliases),
        "| Status:", version.status
    )
