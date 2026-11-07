"""
LAB 6 - MODEL TRAINING AND REGISTRY
"""

from pathlib import Path
import time

import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

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


MODEL_NAME = "LoanDefaultClassifier"


def train_and_register():

    print("=" * 60)

    print(
        "LAB 6 - MODEL TRAINING "
        "AND REGISTRY"
    )

    print("=" * 60)

    # Load processed data
    X_train = np.load(
        PROCESSED_DIR /
        "X_train_final.npy"
    )

    X_test = np.load(
        PROCESSED_DIR /
        "X_test_final.npy"
    )

    y_train = np.load(
        PROCESSED_DIR /
        "y_train.npy"
    )

    y_test = np.load(
        PROCESSED_DIR /
        "y_test.npy"
    )

    print(
        "Training data:",
        X_train.shape
    )

    print(
        "Testing data :",
        X_test.shape
    )

    # MLflow database
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    experiment_name = (
        "Loan_Default_Registry"
    )

    experiment = (
        mlflow.get_experiment_by_name(
            experiment_name
        )
    )

    if experiment is None:

        experiment_id = (
            mlflow.create_experiment(
                experiment_name
            )
        )

    else:

        experiment_id = (
            experiment.experiment_id
        )

    # Models
    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=12,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                random_state=42
            ),

        "Gradient Boosting":
            HistGradientBoostingClassifier(
                random_state=42
            )
    }

    results = []

    for model_name, model in models.items():

        print("\n" + "-" * 60)

        print(
            "Training:",
            model_name
        )

        print("-" * 60)

        start_time = time.time()

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.time() - start_time
        )

        # Predictions
        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        # Metrics
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        # MLflow run
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=
                f"Lab6_{model_name.replace(' ', '_')}"
        ) as run:

            # Parameters
            mlflow.log_params({

                "model":
                    model_name,

                "random_state":
                    42,

                "test_size":
                    0.20,

                "SMOTE":
                    True,

                "feature_count":
                    X_train.shape[1]
            })

            # Metrics
            mlflow.log_metrics({

                "accuracy":
                    accuracy,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1_score":
                    f1,

                "roc_auc":
                    roc_auc,

                "training_time_seconds":
                    training_time
            })

            # Required trusted types for your MLflow version
            trusted_types = [

                "sklearn.tree._tree.Tree",

                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ]

            # Save model to MLflow
            mlflow.sklearn.log_model(

                model,

                name="model",

                skops_trusted_types=
                    trusted_types
            )

            results.append({

                "run_id":
                    run.info.run_id,

                "model":
                    model_name,

                "accuracy":
                    accuracy,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1_score":
                    f1,

                "roc_auc":
                    roc_auc,

                "training_time":
                    training_time
            })

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results
    )

    # Select model based on F1
    selected = results_df.loc[
        results_df["f1_score"].idxmax()
    ]

    print("\nSelected model for registry:")

    print(
        selected["model"]
    )

    run_id = selected["run_id"]

    model_uri = (
        f"runs:/{run_id}/model"
    )

    # Register model
    print(
        "\nRegistering model..."
    )

    registered_model = (
        mlflow.register_model(

            model_uri=model_uri,

            name=MODEL_NAME
        )
    )

    print(
        f"Model registered successfully."
    )

    print(
        f"Model name: {MODEL_NAME}"
    )

    print(
        f"Version: {registered_model.version}"
    )

    # Save results
    results_df.to_csv(

        OUTPUT_DIR /
        "lab6_registry_results.csv",

        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        "outputs/lab6_registry_results.csv"
    )

    print(
        "\nLab 6 training and "
        "registry completed successfully!"
    )


if __name__ == "__main__":

    train_and_register()