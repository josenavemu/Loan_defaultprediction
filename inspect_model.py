import joblib
from pathlib import Path

root = Path.cwd()

model = joblib.load(root / "models" / "best_model.pkl")
ohe = joblib.load(root / "models" / "ohe.pkl")

print("MODEL DETAILS")
print("Model type:", type(model).__name__)
print("Expected features:", getattr(model, "n_features_in_", "Not available"))
print("Classes:", getattr(model, "classes_", "Not available"))

print("\nENCODER DETAILS")
print("Categorical columns:", list(ohe.feature_names_in_))
print("Encoded feature count:", len(ohe.get_feature_names_out()))
print("Encoder categories:")
for column, categories in zip(ohe.feature_names_in_, ohe.categories_):
    print(column, ":", list(categories))
