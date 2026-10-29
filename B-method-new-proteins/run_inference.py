import os
import sys
import dill
import pandas as pd

# Ensure custom classes (like RHCF) can be imported
sys.path.append(os.path.abspath("../"))
from utils import RHCF

# --- 1. Load the trained pipeline ---
model_type = "XGB"  # or "LR"
model_path = "./B/XGB_full_pipeline.pkl"

with open(model_path, "rb") as f:
    bundle = dill.load(f)

# --- 2. Load the new Excel dataset ---
new_file_path = "./B-method-new-proteins/dataset_features/total.xlsx"
df_new = pd.read_excel(new_file_path, index_col=0).set_index("PDB Code(s)")

# 3. Align columns to match the 2,720 training features exactly
df_aligned = df_new.reindex(columns=bundle["feature_names"])

# 4. Extract feature matrix X (now guaranteed to have 2,720 features)
X_new = df_aligned.values

# 5. Run sequential pipeline
X_imputed = bundle["imputer"].transform(X_new)        # Expects 2720 -> Outputs 2720
X_rhcf = bundle["rhcf"].transform(X_imputed)           # Expects 2720 -> Outputs 2342

if bundle["scaler"] is not None:
    X_rhcf = bundle["scaler"].transform(X_rhcf)

# 6. Predict
predictions = bundle["model"].predict(X_rhcf)

# --- 6. Save Predictions back to Excel ---
df_results = pd.DataFrame({
    "PDB_Code": df_new.index,
    "Predicted_Em_mV": predictions
}, index=df_new.index)

output_excel_path = f"{model_type}_new_predictions.csv"
df_results.to_csv(output_excel_path)

print(f"Done! Predictions saved to {output_excel_path}")