import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPORT = (
    PROJECT_ROOT / "outputs" / "drift" / "lab12_drift_report.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "monitoring"


def evaluate_report(report):
    required_fields = [
        "moderate_drift_features",
        "significant_drift_features",
        "average_psi",
        "feature_count"
    ]

    for field in required_fields:
        if field not in report:
            raise ValueError(f"Missing field in Lab 12 report: {field}")

    moderate = report["moderate_drift_features"]
    significant = report["significant_drift_features"]
    average_psi = report["average_psi"]
    feature_count = report["feature_count"]

    if not all(isinstance(value, int) and value >= 0
               for value in [moderate, significant, feature_count]):
        raise ValueError("Drift counts and feature count must be non-negative integers.")

    if feature_count == 0:
        raise ValueError("Feature count must be greater than zero.")

    if not isinstance(average_psi, (int, float)) or average_psi < 0:
        raise ValueError("Average PSI must be a non-negative number.")

    if significant > 0:
        status = "CRITICAL"
        action = "Review the affected features and investigate the data source before relying on predictions."
        alert = True

    elif moderate > 0:
        status = "WARNING"
        action = "Review moderately drifted features and continue monitoring."
        alert = True

    else:
        status = "HEALTHY"
        action = "No significant or moderate drift detected. Continue routine monitoring."
        alert = False

    return {
        "lab": 13,
        "project": "Loan Default Prediction",
        "monitoring_status": status,
        "alert_triggered": alert,
        "moderate_drift_features": moderate,
        "significant_drift_features": significant,
        "feature_count": feature_count,
        "average_psi": average_psi,
        "recommended_action": action,
        "created_at_utc": datetime.now(timezone.utc).isoformat()
    }


def run_monitoring_trigger(report_path=DEFAULT_REPORT, output_dir=OUTPUT_DIR):
    report_path = Path(report_path)
    output_dir = Path(output_dir)

    if not report_path.exists():
        raise FileNotFoundError(
            f"Lab 12 report not found: {report_path}. Run Lab 12 first."
        )

    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    decision = evaluate_report(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / "lab13_monitoring_decision.json"

    with open(decision_path, "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=4)

    print("\n========== LAB 13 MONITORING TRIGGER ==========")
    print(f"Monitoring status: {decision['monitoring_status']}")
    print(f"Alert triggered: {decision['alert_triggered']}")
    print(f"Significant drift features: {decision['significant_drift_features']}")
    print(f"Moderate drift features: {decision['moderate_drift_features']}")
    print(f"Average PSI: {decision['average_psi']}")
    print(f"Recommended action: {decision['recommended_action']}")
    print(f"Decision report saved: {decision_path}")
    print("LAB 13 MONITORING CHECK COMPLETED")

    return decision


if __name__ == "__main__":
    run_monitoring_trigger()
