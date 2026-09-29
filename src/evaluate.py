import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    classification_report
)


def run_evaluation():
    print("Starting Model Evaluation...")

    os.makedirs("outputs", exist_ok=True)

    # 1. Load test data
    X_test = np.load(
        "data/processed/X_test_final.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    print("Test data:", X_test.shape)
    print("Test target:", y_test.shape)

    # 2. Model file names
    model_files = {
        "Logistic Regression":
            "models/logistic_regression.pkl",

        "Decision Tree":
            "models/decision_tree.pkl",

        "Random Forest":
            "models/random_forest.pkl",

        "Gradient Boosting":
            "models/gradient_boosting.pkl"
    }

    results = []
    predictions = {}
    probabilities = {}

    # 3. Evaluate every model
    for name, file_path in model_files.items():

        if not os.path.exists(file_path):
            print(
                f"Skipping {name}: "
                f"{file_path} not found."
            )
            continue

        print("\nEvaluating:", name)

        model = joblib.load(file_path)

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        predictions[name] = y_pred
        probabilities[name] = y_prob

        results.append({
            "Model": name,
            "Accuracy":
                accuracy_score(
                    y_test,
                    y_pred
                ),

            "Precision":
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "F1-score":
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "ROC-AUC":
                roc_auc_score(
                    y_test,
                    y_prob
                )
        })

        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Default",
                "Default"
            ],
            zero_division=0
        )

        print("\n", name)
        print(report)

        # Confusion matrix
        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=[
                "No Default",
                "Default"
            ],
            cmap="Blues",
            colorbar=False,
            ax=ax
        )

        ax.set_title(
            name + " - Confusion Matrix"
        )

        plt.tight_layout()

        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        plt.savefig(
            f"outputs/{safe_name}_confusion_matrix.png",
            dpi=300
        )

        plt.close()

    # 4. Save model comparison
    results_df = (
        pd.DataFrame(results)
        .set_index("Model")
        .sort_values(
            "F1-score",
            ascending=False
        )
    )

    results_df.round(4).to_csv(
        "outputs/model_comparison.csv"
    )

    print("\nModel Comparison:")
    print(results_df.round(4))

    # 5. Model comparison graph
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ]

    results_df[metrics].plot(
        kind="bar",
        figsize=(11, 5)
    )

    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        "outputs/model_comparison.png",
        dpi=300
    )

    plt.close()

    # 6. ROC curves
    plt.figure(figsize=(7, 6))

    for name, y_prob in probabilities.items():

        fpr, tpr, _ = roc_curve(
            y_test,
            y_prob
        )

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random guess"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "outputs/roc_curves.png",
        dpi=300
    )

    plt.close()

    # 7. Best model by F1-score
    if not results_df.empty:

        best_name = results_df[
            "F1-score"
        ].idxmax()

        print(
            "\nBest model by F1-score:",
            best_name
        )

        best_report = classification_report(
            y_test,
            predictions[best_name],
            target_names=[
                "No Default",
                "Default"
            ],
            zero_division=0
        )

        with open(
            "outputs/best_model_report.txt",
            "w"
        ) as f:
            f.write(
                f"Best model by F1-score: "
                f"{best_name}\n\n"
            )

            f.write(best_report)

        # Save the selected model
        source_model = model_files[best_name]

        joblib.dump(
            joblib.load(source_model),
            "models/best_model.pkl"
        )

        print(
            "Saved:",
            "models/best_model.pkl"
        )

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    run_evaluation()
