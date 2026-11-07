import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lab9_reliability import (
    load_model,
    predict_safely,
    PipelineError
)


def run_lab9():

    report = {
        "lab": "Lab 9 - Reliability and Failure Handling",
        "timestamp": datetime.now().isoformat(),
        "status": "FAILED",
        "checks": {},
        "errors": []
    }

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "lab9_reliability_report.json"

    try:
        print("=" * 60)
        print("LAB 9 - RELIABILITY AND FAILURE HANDLING")
        print("=" * 60)

        model_path = PROJECT_ROOT / "models" / "best_model.pkl"
        test_path = PROJECT_ROOT / "data" / "processed" / "X_test_final.npy"

        if not model_path.exists():
            raise PipelineError("Best model artifact is missing.")

        if not test_path.exists():
            raise PipelineError("Test dataset is missing.")

        print("\n1. Checking model artifact...")
        model = load_model(model_path)
        report["checks"]["model_loading"] = "PASSED"
        print("Model loaded:", type(model).__name__)

        print("\n2. Loading test dataset...")
        X_test = np.load(test_path, allow_pickle=False)
        report["checks"]["test_data_loading"] = "PASSED"
        print("Test data shape:", X_test.shape)

        print("\n3. Validating input features...")
        sample = X_test[:100]
        predictions = predict_safely(model, sample)
        report["checks"]["feature_validation"] = "PASSED"
        print("Feature validation successful.")

        print("\n4. Testing model predictions...")
        report["checks"]["prediction"] = "PASSED"
        report["prediction_count"] = int(len(predictions))
        report["sample_predictions"] = predictions[:10].tolist()
        print("Predictions generated:", len(predictions))

        report["status"] = "SUCCESS"

    except Exception as exc:
        report["errors"].append(str(exc))
        print("\nLAB 9 FAILED:", exc)

    finally:
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("\nReport saved to:", report_path)
        print("Final status:", report["status"])

    if report["status"] != "SUCCESS":
        raise SystemExit(1)


if __name__ == "__main__":
    run_lab9()
