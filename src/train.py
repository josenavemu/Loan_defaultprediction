import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)


def run_training():
    print("Starting Model Training...")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load processed data
    X_train = np.load(
        "data/processed/X_train_final.npy"
    )

    y_train = np.load(
        "data/processed/y_train.npy"
    )

    print("Training data:", X_train.shape)
    print("Training target:", y_train.shape)

    # 2. Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=42
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            random_state=42
        )
    }

    results = []

    # 3. Train all models
    for name, model in models.items():

        print("\nTraining:", name)

        start = time.time()

        model.fit(
            X_train,
            y_train
        )

        training_time = round(
            time.time() - start,
            2
        )

        # Create a safe file name
        file_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        joblib.dump(
            model,
            f"models/{file_name}.pkl"
        )

        results.append({
            "Model": name,
            "Train time (s)": training_time
        })

        print(
            name,
            "trained successfully in",
            training_time,
            "seconds"
        )

    # 4. Save training information
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "outputs/training_results.csv",
        index=False
    )

    print("\nAll models trained successfully.")


if __name__ == "__main__":
    run_training()
