import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException

from api.schemas import LoanApplication
from api.model_loader import load_artifacts


app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default using a trained ML model.",
    version="1.0.0"
)

try:
    artifacts = load_artifacts()

    model = artifacts["model"]
    imputer = artifacts["imputer"]
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]

    numerical_columns = [
        "loan_amount",
        "term",
        "property_value",
        "income",
        "Credit_Score",
        "LTV",
        "dtir1"
    ]

    categorical_columns = list(encoder.feature_names_in_)

except Exception as exc:
    raise RuntimeError(
        f"API initialization failed: {exc}"
    ) from exc


@app.get("/")
def home():

    return {
        "message": "Loan Default Prediction API is running",
        "status": "healthy"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "expected_features": 51
    }


@app.post("/predict")
def predict(application: LoanApplication):

    try:
        input_data = pd.DataFrame(
            [application.model_dump(by_alias=True)]
        )

        # Numerical preprocessing
        numeric_data = input_data[
            numerical_columns
        ]

        numeric_data = imputer.transform(
            numeric_data
        )

        numeric_data = scaler.transform(
            numeric_data
        )

        # Categorical preprocessing
        categorical_data = input_data[
            categorical_columns
        ].fillna("Missing")

        categorical_data = encoder.transform(
            categorical_data
        )

        # Combine numerical and categorical features
        final_data = np.hstack(
            [numeric_data, categorical_data]
        )

        if final_data.shape[1] != 51:
            raise ValueError(
                "Preprocessing did not produce 51 features."
            )

        if not np.isfinite(final_data).all():
            raise ValueError(
                "Processed input contains invalid values."
            )

        prediction = int(
            model.predict(final_data)[0]
        )

        probabilities = model.predict_proba(
            final_data
        )[0]

        return {
            "predicted_class": prediction,
            "class_probabilities": {
                str(label): round(float(probability), 4)
                for label, probability in zip(
                    model.classes_,
                    probabilities
                )
            },
            "model": type(model).__name__,
            "features_used": int(final_data.shape[1])
        }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(exc)}"
        )

