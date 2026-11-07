
"""
LAB 7 - ML PIPELINE ENGINEERING

Integrates data validation, preprocessing, training,
evaluation, inference, and artifact tracking.
"""

from pathlib import Path
import subprocess
import sys
import json
import time
import hashlib

import numpy as np
import joblib


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# Pipeline execution details
pipeline_results = []


def run_stage(stage_name, script_path):

    print("\n" + "=" * 60)
    print("RUNNING:", stage_name)
    print("=" * 60)

    start_time = time.time()

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True
    )

    elapsed = round(time.time() - start_time, 2)

    pipeline_results.append({
        "stage": stage_name,
        "status": "SUCCESS",
        "execution_time_seconds": elapsed
    })

    print(f"\n{stage_name} completed in {elapsed} seconds.")


def calculate_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def run_inference():

    print("\n" + "=" * 60)
    print("PIPELINE INFERENCE")
    print("=" * 60)

    model_path = MODEL_DIR / "best_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "Best model artifact not found."
        )

    model = joblib.load(model_path)

    X_test = np.load(
        DATA_DIR / "X_test_final.npy"
    )

    # Predict a small sample from the test dataset
    sample = X_test[:5]

    predictions = model.predict(sample)

    inference_results = []

    for index, prediction in enumerate(predictions):

        result = {
            "sample": index + 1,
            "prediction": int(prediction),
            "class": (
                "Default" if prediction == 1
                else "No Default"
            )
        }

        inference_results.append(result)

        print(result)

    inference_path = (
        OUTPUT_DIR / "lab7_inference_results.json"
    )

    with open(
        inference_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            inference_results,
            file,
            indent=4
        )

    print("\nInference completed successfully.")


def generate_pipeline_report():

    print("\n" + "=" * 60)
    print("GENERATING PIPELINE REPORT")
    print("=" * 60)

    artifact_files = [
        DATA_DIR / "X_train_final.npy",
        DATA_DIR / "X_test_final.npy",
        DATA_DIR / "y_train.npy",
        DATA_DIR / "y_test.npy",
        MODEL_DIR / "best_model.pkl",
        OUTPUT_DIR / "model_comparison.csv",
        OUTPUT_DIR / "lab7_inference_results.json"
    ]

    artifacts = {}

    for file_path in artifact_files:

        if file_path.exists():

            artifacts[file_path.name] = {
                "size_bytes": file_path.stat().st_size,
                "sha256": calculate_hash(file_path)
            }

        else:
            raise FileNotFoundError(
                f"Required artifact missing: {file_path}"
            )

    report = {
        "experiment": "Lab 7 - ML Pipeline Engineering",
        "overall_status": "SUCCESS",
        "stages": pipeline_results,
        "artifacts": artifacts
    }

    report_path = (
        OUTPUT_DIR / "lab7_pipeline_report.json"
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

    print("Pipeline report saved:", report_path)


def run_pipeline():

    print("=" * 60)
    print("LAB 7 - ML PIPELINE ENGINEERING")
    print("=" * 60)

    total_start = time.time()

    try:

        # Stage 1: Data validation
        run_stage(
            "Data Validation",
            SRC_DIR / "validate_data.py"
        )

        # Stage 2: Preprocessing
        run_stage(
            "Preprocessing",
            SRC_DIR / "preprocess_pipeline.py"
        )

        # Stage 3: Output validation
        run_stage(
            "Preprocessing Output Validation",
            SRC_DIR / "validate_outputs.py"
        )

        # Stage 4: Model training
        run_stage(
            "Model Training",
            SRC_DIR / "train.py"
        )

        # Stage 5: Model evaluation
        run_stage(
            "Model Evaluation",
            SRC_DIR / "evaluate.py"
        )

        # Stage 6: Inference
        run_inference()

        # Stage 7: Artifact tracking
        generate_pipeline_report()

        total_time = round(
            time.time() - total_start, 2
        )

        print("\n" + "=" * 60)
        print("LAB 7 COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print("Total execution time:", total_time, "seconds")

    except Exception as error:

        print("\nPIPELINE FAILED!")
        print("Error:", error)

        raise


if __name__ == "__main__":
    run_pipeline()
