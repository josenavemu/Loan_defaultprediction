from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


def load_artifacts():

    required_files = {
        "model": MODEL_DIR / "best_model.pkl",
        "imputer": MODEL_DIR / "imputer.pkl",
        "scaler": MODEL_DIR / "scaler.pkl",
        "encoder": MODEL_DIR / "ohe.pkl"
    }

    artifacts = {}

    for name, path in required_files.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

        artifacts[name] = joblib.load(path)

    if artifacts["model"].n_features_in_ != 51:
        raise ValueError(
            "Model does not accept the expected 51 features."
        )

    return artifacts
