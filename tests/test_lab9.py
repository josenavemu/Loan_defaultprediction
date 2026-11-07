import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lab9_reliability import (
    validate_features,
    load_model,
    predict_safely,
    PipelineError
)


class TestLoanDefaultReliability(unittest.TestCase):

    def test_valid_features(self):
        data = np.zeros((5, 51))
        result = validate_features(data)
        self.assertEqual(result.shape, (5, 51))

    def test_wrong_feature_count(self):
        data = np.zeros((5, 50))
        with self.assertRaises(PipelineError):
            validate_features(data)

    def test_empty_input(self):
        data = np.empty((0, 51))
        with self.assertRaises(PipelineError):
            validate_features(data)

    def test_one_dimensional_input(self):
        data = np.zeros(51)
        with self.assertRaises(PipelineError):
            validate_features(data)

    def test_nan_input(self):
        data = np.zeros((2, 51))
        data[0, 0] = np.nan
        with self.assertRaises(PipelineError):
            validate_features(data)

    def test_infinite_input(self):
        data = np.zeros((2, 51))
        data[0, 0] = np.inf
        with self.assertRaises(PipelineError):
            validate_features(data)

    def test_missing_model(self):
        missing_path = PROJECT_ROOT / "models" / "missing_model.pkl"
        with self.assertRaises(PipelineError):
            load_model(missing_path)

    def test_real_model_prediction(self):
        model = load_model()
        data = np.zeros((2, 51))
        predictions = predict_safely(model, data)
        self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
