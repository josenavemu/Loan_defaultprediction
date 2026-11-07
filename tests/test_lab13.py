import unittest

from src.lab13_monitoring_trigger import evaluate_report


class TestLab13MonitoringTrigger(unittest.TestCase):

    def test_healthy_status(self):
        report = {
            "moderate_drift_features": 0,
            "significant_drift_features": 0,
            "average_psi": 0.003528,
            "feature_count": 51
        }

        result = evaluate_report(report)

        self.assertEqual(result["monitoring_status"], "HEALTHY")
        self.assertFalse(result["alert_triggered"])

    def test_warning_status(self):
        report = {
            "moderate_drift_features": 2,
            "significant_drift_features": 0,
            "average_psi": 0.15,
            "feature_count": 51
        }

        result = evaluate_report(report)

        self.assertEqual(result["monitoring_status"], "WARNING")
        self.assertTrue(result["alert_triggered"])

    def test_critical_status(self):
        report = {
            "moderate_drift_features": 1,
            "significant_drift_features": 3,
            "average_psi": 0.30,
            "feature_count": 51
        }

        result = evaluate_report(report)

        self.assertEqual(result["monitoring_status"], "CRITICAL")
        self.assertTrue(result["alert_triggered"])

    def test_missing_field(self):
        report = {
            "moderate_drift_features": 0,
            "significant_drift_features": 0,
            "average_psi": 0.003528
        }

        with self.assertRaises(ValueError):
            evaluate_report(report)


if __name__ == "__main__":
    unittest.main()
