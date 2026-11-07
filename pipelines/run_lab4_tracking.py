import os
import sys
import time
import subprocess
import joblib
import numpy as np
import pandas as pd

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    print("MLflow is not installed.")
    print("Run:")
    print("pip install mlflow")
    sys.exit(1)

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


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

os.chdir(PROJECT_ROOT)


def run_preprocessing():
    print("\n" + "=" * 60)
    print("STEP 1 - PREPROCESSING")
    print("=" * 60)

    script_path = os.path.join(
        PROJECT_ROOT,
        "src",
        "preprocess.py"
    )

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        sys.exit(result.returncode)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        )
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(
            y_test,
            y_prob
        )

    return metrics


def main():

    print("=" * 60)
    print("LAB 4 - MLflow EXPERIMENT TRACKING")
    print("=" * 60)

    os.makedirs(
        "outputs/mlflow",
        exist_ok=True
    )

    # ---------------------------------------------------------
    # STEP 1: Run preprocessing
    # ---------------------------------------------------------
    run_preprocessing()

    # ---------------------------------------------------------
    # STEP 2: Load processed data
    # ---------------------------------------------------------
    X_train = np.load(
        "data/processed/X_train_final.npy"
    )

    y_train = np.load(
        "data/processed/y_train.npy"
    )

    X_test = np.load(
        "data/processed/X_test_final.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    print("\nTraining shape:", X_train.shape)
    print("Testing shape :", X_test.shape)

    # ---------------------------------------------------------
    # STEP 3: MLflow setup
    # ---------------------------------------------------------
    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        "Loan_Default_Experiments"
    )

    # ---------------------------------------------------------
    # STEP 4: Define experiments
    # ---------------------------------------------------------
    experiments = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            {
                "model_type": "LogisticRegression",
                "max_iter": 1000
            }
        ),

        (
            "Decision Tree",
            DecisionTreeClassifier(
                max_depth=12,
                random_state=42
            ),
            {
                "model_type": "DecisionTree",
                "max_depth": 12
            }
        ),

        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                random_state=42
            ),
            {
                "model_type": "RandomForest",
                "n_estimators": 100,
                "random_state": 42
            }
        ),

        (
            "Gradient Boosting",
            HistGradientBoostingClassifier(
                random_state=42
            ),
            {
                "model_type": "HistGradientBoosting",
                "random_state": 42
            }
        )
    ]

    results = []

    # ---------------------------------------------------------
    # STEP 5: Train and track each experiment
    # ---------------------------------------------------------
    for name, model, params in experiments:

        print("\n" + "-" * 60)
        print("Running experiment:", name)
        print("-" * 60)

        with mlflow.start_run(
            run_name=name.replace(" ", "_")
        ):

            start_time = time.time()

            # Parameters
            mlflow.log_params(params)

            mlflow.log_param(
                "test_size",
                0.2
            )

            mlflow.log_param(
                "random_state",
                42
            )

            mlflow.log_param(
                "smote",
                True
            )

            mlflow.log_param(
                "preprocessing",
                "median_imputation + StandardScaler + OneHotEncoder"
            )

            # Dataset information
            mlflow.log_param(
                "train_rows",
                X_train.shape[0]
            )

            mlflow.log_param(
                "test_rows",
                X_test.shape[0]
            )

            mlflow.log_param(
                "feature_count",
                X_train.shape[1]
            )

            # Training
            model.fit(
                X_train,
                y_train
            )

            training_time = (
                time.time() - start_time
            )

            mlflow.log_metric(
                "training_time_seconds",
                training_time
            )

            # Evaluation
            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model

            mlflow.sklearn.log_model(model,artifact_path="model",skops_trusted_types=["sklearn.tree._tree.Tree","sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"])

            

            # Save local model
            safe_name = (
                name.lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            local_model_path = (
                f"models/{safe_name}_mlflow.pkl"
            )

            os.makedirs(
                "models",
                exist_ok=True
            )

            joblib.dump(
                model,
                local_model_path
            )

            # Save result
            result_row = {
                "Model": name,
                **metrics,
                "training_time_seconds":
                    training_time
            }

            results.append(result_row)

            print("Metrics:")
            for metric, value in metrics.items():
                print(
                    f"  {metric}: {value:.4f}"
                )

    # ---------------------------------------------------------
    # STEP 6: Save experiment summary
    # ---------------------------------------------------------
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "outputs/mlflow/experiment_results.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print(
        results_df.round(4).to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(
        "outputs/mlflow/experiment_results.csv"
    )
    print("mlflow.db")
    print("MLflow model artifacts")

    print("\n" + "=" * 60)
    print("LAB 4 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nTo open the MLflow UI, run:")
    print("mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
