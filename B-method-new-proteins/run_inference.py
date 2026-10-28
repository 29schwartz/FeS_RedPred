import joblib
import pandas as pd
import os
from xgboost import XGBRegressor

MODEL_JSON = "xgboost_model.json"
PREPROCESSOR_PATH = "xgboost_preprocessor.joblib"           # Path to your saved model file
FEATURES_PATH = "./B-method-new-proteins/dataset_features/total.xlsx"   # Features from Step 1
PREDICTIONS_OUTPUT = "redox_predictions.csv"

# 1. Load extracted features
df_new = pd.read_excel(FEATURES_PATH)
pdb_ids = df_new["PDB_ID"] if "PDB_ID" in df_new.columns else df_new.index

# Select feature columns (excluding metadata)
feature_cols = [c for c in df_new.columns if c not in ["PDB_ID", "Experimental_Em", "Redox_Potential"]]
X_new = df_new[feature_cols]

# 2. Run inference based on saved model format
if os.path.exists(MODEL_JSON) and os.path.exists(PREPROCESSOR_PATH):
    # Load XGBoost native JSON model + preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    xgb_model = XGBRegressor()
    xgb_model.load_model(MODEL_JSON)

    # Transform features through pipeline (scaling/feature selection) and predict
    X_transformed = preprocessor.transform(X_new)
    predictions = xgb_model.predict(X_transformed)

else:
    # Fallback for joblib pipelines (RandomForest, SVR, etc.)
    pipeline_file = [f for f in os.listdir(".") if f.endswith("_pipeline.joblib")][0]
    pipeline = joblib.load(pipeline_file)
    predictions = pipeline.predict(X_new)

# 3. Save results
df_results = pd.DataFrame({
    "PDB_ID": pdb_ids,
    "Predicted_Em_mV": predictions
})

df_results.to_csv(PREDICTIONS_OUTPUT, index=False)
print("\n--- Predictions Complete ---")
print(df_results)