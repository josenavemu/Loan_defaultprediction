import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "drift"

REFERENCE_PATH = DATA_DIR / "X_train_final.npy"
CURRENT_PATH = DATA_DIR / "X_test_final.npy"
FEATURE_NAMES_PATH = DATA_DIR / "feature_names.json"

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
EPSILON = 1e-6


def calculate_psi(reference, current):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if reference.size == 0 or current.size == 0:
        raise ValueError("Reference and current data cannot be empty.")

    unique_values = np.unique(reference)

    if unique_values.size == 1:
        ref_bins = np.array([
            np.sum(reference == unique_values[0]),
            np.sum(reference != unique_values[0])
        ], dtype=float)

        cur_bins = np.array([
            np.sum(current == unique_values[0]),
            np.sum(current != unique_values[0])
        ], dtype=float)

    elif unique_values.size == 2:
        threshold = (unique_values[0] + unique_values[1]) / 2

        ref_bins = np.array([
            np.sum(reference <= threshold),
            np.sum(reference > threshold)
        ], dtype=float)

        cur_bins = np.array([
            np.sum(current <= threshold),
            np.sum(current > threshold)
        ], dtype=float)

    else:
        edges = np.unique(
            np.quantile(reference, np.linspace(0, 1, 11))
        )

        if edges.size < 2:
            raise ValueError("Unable to create PSI bins.")

        edges[0] = -np.inf
        edges[-1] = np.inf

        ref_bins, _ = np.histogram(reference, bins=edges)
        cur_bins, _ = np.histogram(current, bins=edges)

        ref_bins = ref_bins.astype(float)
        cur_bins = cur_bins.astype(float)

    ref_pct = (ref_bins + EPSILON) / (
        np.sum(ref_bins) + EPSILON * len(ref_bins)
    )
    cur_pct = (cur_bins + EPSILON) / (
        np.sum(cur_bins) + EPSILON * len(cur_bins)
    )

    psi = np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    )

    return float(psi)


def classify_drift(psi):
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "low"


def run_drift_detection():
    for path in [REFERENCE_PATH, CURRENT_PATH, FEATURE_NAMES_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    reference = np.load(REFERENCE_PATH)
    current = np.load(CURRENT_PATH)

    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    if reference.ndim != 2 or current.ndim != 2:
        raise ValueError("Both datasets must be two-dimensional.")

    if reference.shape[1] != current.shape[1]:
        raise ValueError("Reference and current feature counts differ.")

    if reference.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match the data columns.")

    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Data contains NaN or infinite values.")

    feature_results = []

    for index, feature_name in enumerate(feature_names):
        psi = calculate_psi(
            reference[:, index],
            current[:, index]
        )

        feature_results.append({
            "feature": feature_name,
            "psi": round(psi, 6),
            "drift_level": classify_drift(psi)
        })

    feature_results.sort(
        key=lambda item: item["psi"],
        reverse=True
    )

    significant = [
        item for item in feature_results
        if item["drift_level"] == "significant"
    ]

    moderate = [
        item for item in feature_results
        if item["drift_level"] == "moderate"
    ]

    average_psi = float(
        np.mean([item["psi"] for item in feature_results])
    )

    report = {
        "lab": 12,
        "project": "Loan Default Prediction",
        "monitoring_method": "Population Stability Index (PSI)",
        "reference_dataset": str(REFERENCE_PATH.relative_to(PROJECT_ROOT)),
        "current_dataset": str(CURRENT_PATH.relative_to(PROJECT_ROOT)),
        "reference_rows": int(reference.shape[0]),
        "current_rows": int(current.shape[0]),
        "feature_count": int(reference.shape[1]),
        "thresholds": {
            "low": "PSI < 0.10",
            "moderate": "0.10 <= PSI < 0.25",
            "significant": "PSI >= 0.25"
        },
        "moderate_drift_features": len(moderate),
        "significant_drift_features": len(significant),
        "drift_detected": len(significant) > 0,
        "average_psi": round(average_psi, 6),
        "top_10_drifted_features": feature_results[:10],
        "feature_results": feature_results,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation_note": (
            "Training data was SMOTE-balanced while test data was not. "
            "Some distribution differences may reflect this preprocessing "
            "choice rather than genuine production drift. This comparison "
            "is exploratory and does not prove model performance degradation."
        )
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT_DIR / "lab12_drift_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n========== LAB 12 DRIFT DETECTION ==========")
    print(f"Reference rows: {reference.shape[0]}")
    print(f"Current rows: {current.shape[0]}")
    print(f"Features checked: {reference.shape[1]}")
    print(f"Moderate drift features: {len(moderate)}")
    print(f"Significant drift features: {len(significant)}")
    print(f"Average PSI: {average_psi:.6f}")
    print(f"Drift detected: {report['drift_detected']}")
    print("\nTop 10 drifted features:")

    for item in feature_results[:10]:
        print(
            f"{item['feature']}: "
            f"PSI={item['psi']}, "
            f"Level={item['drift_level']}"
        )

    print(f"\nReport saved to: {report_path}")
    print("LAB 12 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_drift_detection()
