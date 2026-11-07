from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

EXPECTED_FEATURES = 51


class PipelineError(Exception):
    """Custom exception for reliability failures."""
    pass


def validate_features(data):
    if data is None:
        raise PipelineError("Input data is missing.")

    data = np.asarray(data)

    if data.ndim != 2:
        raise PipelineError("Input must be a 2D array.")

    if data.shape[0] == 0:
        raise PipelineError("Input contains no rows.")

    if data.shape[1] != EXPECTED_FEATURES:
        raise PipelineError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"but received {data.shape[1]}."
        )

    if not np.issubdtype(data.dtype, np.number):
        raise PipelineError("Input must contain numeric values.")

    if not np.isfinite(data).all():
        raise PipelineError("Input contains NaN or infinite values.")

    return data


def load_model(model_path=MODEL_PATH):
    model_path = Path(model_path)

    if not model_path.exists():
        raise PipelineError(
            f"Model file not found: {model_path}"
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise PipelineError(
            f"Model loading failed: {exc}"
        ) from exc

    if getattr(model, "n_features_in_", None) != EXPECTED_FEATURES:
        raise PipelineError(
            "Loaded model does not accept exactly 51 features."
        )

    return model


def predict_safely(model, data):
    data = validate_features(data)

    try:
        predictions = model.predict(data)
    except Exception as exc:
        raise PipelineError(
            f"Prediction failed: {exc}"
        ) from exc

    return predictions
