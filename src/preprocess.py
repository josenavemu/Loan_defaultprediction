import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # Create required folders
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load the raw data
    # Change this path if your loan dataset is stored somewhere else.
    data_path = r"C:\Users\user\Downloads\Loan_Default.csv (1)\Loan_Default.csv"
    df = pd.read_csv(data_path)

    print("Original dataset shape:", df.shape)

    # 2. Drop useless / leaky columns and duplicates
    drop_columns = [
        "ID",
        "year",
        "rate_of_interest",
        "Interest_rate_spread",
        "Upfront_charges"
    ]

    df = df.drop(columns=drop_columns, errors="ignore")

    before = len(df)
    df = df.drop_duplicates()
    print("Duplicate rows removed:", before - len(df))

    # 3. Separate features and target
    X = df.drop("Status", axis=1)
    y = df["Status"]

    # 4. Split first
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # 5. Separate numeric and categorical columns
    cat_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    num_cols = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    print("Numerical columns:", num_cols)
    print("Categorical columns:", cat_cols)

    # 6. Correlation matrix
    # Calculated on training data only.
    corr_matrix = X_train[num_cols].join(y_train).corr()

    corr_matrix.to_csv("outputs/correlation_matrix.csv")

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Correlation Matrix - Loan Default Dataset")
    plt.tight_layout()
    plt.savefig("outputs/correlation_matrix.png", dpi=300)
    plt.close()

    # Correlation with target
    target_corr = (
        corr_matrix["Status"]
        .drop("Status")
        .sort_values()
    )

    target_corr.to_csv("outputs/correlation_with_status.csv")

    plt.figure(figsize=(8, 6))
    target_corr.plot(kind="barh")
    plt.title("Correlation with Status")
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.savefig("outputs/correlation_with_status.png", dpi=300)
    plt.close()

    # 7. Drop highly correlated numeric features
    THRESHOLD = 0.85

    abs_corr = X_train[num_cols].corr().abs()

    upper = abs_corr.where(
        np.triu(
            np.ones(abs_corr.shape),
            k=1
        ).astype(bool)
    )

    high_corr = [
        column
        for column in upper.columns
        if (upper[column] > THRESHOLD).any()
    ]

    print(
        f"Features with correlation above {THRESHOLD}:",
        high_corr
    )

    X_train = X_train.drop(columns=high_corr)
    X_test = X_test.drop(columns=high_corr)

    num_cols = [
        column
        for column in num_cols
        if column not in high_corr
    ]

    print("Numerical columns kept:", num_cols)

    # 8. Numeric: median imputation, then scaling
    # Fit only on training data.
    imputer = SimpleImputer(strategy="median")

    x_train_num = imputer.fit_transform(
        X_train[num_cols]
    )

    x_test_num = imputer.transform(
        X_test[num_cols]
    )

    scaler = StandardScaler()

    x_train_num = scaler.fit_transform(
        x_train_num
    )

    x_test_num = scaler.transform(
        x_test_num
    )

    # 9. Categorical: fill missing, then one-hot encode
    ohe = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    x_train_cat = ohe.fit_transform(
        X_train[cat_cols].fillna("Missing")
    )

    x_test_cat = ohe.transform(
        X_test[cat_cols].fillna("Missing")
    )

    # 10. Combine numeric and categorical features
    X_train_final = np.hstack(
        (x_train_num, x_train_cat)
    )

    X_test_final = np.hstack(
        (x_test_num, x_test_cat)
    )

    feature_names = (
        num_cols
        + list(ohe.get_feature_names_out(cat_cols))
    )

    print("X_train_final:", X_train_final.shape)
    print("X_test_final :", X_test_final.shape)

    # 11. SMOTE on training data only
    smote = SMOTE(random_state=42)

    X_train_bal, y_train_bal = smote.fit_resample(
        X_train_final,
        y_train
    )

    print("Before SMOTE:")
    print(y_train.value_counts())

    print("\nAfter SMOTE:")
    print(y_train_bal.value_counts())

    # 12. Save processed data
    np.save(
        "data/processed/X_train_final.npy",
        X_train_bal
    )

    np.save(
        "data/processed/X_test_final.npy",
        X_test_final
    )

    np.save(
        "data/processed/y_train.npy",
        y_train_bal.to_numpy(dtype=np.int64)
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.to_numpy(dtype=np.int64)
    )

    # 13. Save preprocessing objects
    joblib.dump(
        imputer,
        "models/imputer.pkl"
    )

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    joblib.dump(
        ohe,
        "models/ohe.pkl"
    )

    # 14. Save metadata
    metadata = {
        "dataset_name": "Loan Default Dataset",
        "target": "Status",
        "train_shape": list(X_train_bal.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols),
        "dropped_columns": drop_columns,
        "high_correlation_features_removed": high_corr,
        "correlation_threshold": THRESHOLD,
        "smote": True,
        "random_state": 42
    }

    with open(
        "data/processed/dataset_metadata.json",
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    # Save feature names
    with open(
        "data/processed/feature_names.json",
        "w"
    ) as f:
        json.dump(feature_names, f, indent=4)

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
