# Purpose: establish regression performance baseline using Morgan Fingerprints to later determine necessity of LLM use (pricier, computationally expensive)
import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
from src.logger import get_console_logger
logger = get_console_logger(__name__)

# HELPER FUNCTIONS:
# Convert a SMILES string to a Morgan fingerprint array in np
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((n_bits,))
        # Generate the Morgan Fingerprint
        fp = morgan_gen.GetFingerprint(mol)
        # Convert to a numpy array (for XGBoost)
        arr = np.zeros((0,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    
    except Exception:
        return np.zeros((n_bits,))

# Load a CSV into a dataframe
def load_and_featurize(file_path: str):
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    logger.info("Generating Morgan Fingerprints...")
    # Stack the individual arrays into a 2D matrix
    X = np.vstack(df['smiles'].apply(smiles_to_fp).values)
    y = df['pIC50'].values
    return X, y

# MAIN FUNCTION:
def train_baseline():
    data_dir = "data/processed"
    model_out_dir = "models/baseline"
    
    # Load data
    X_train, y_train = load_and_featurize(os.path.join(data_dir, "train_herg.csv"))
    X_valid, y_valid = load_and_featurize(os.path.join(data_dir, "valid_herg.csv"))
    X_test,  y_test  = load_and_featurize(os.path.join(data_dir, "test_herg.csv"))
    logger.info(f"Feature matrix shapes - Train: {X_train.shape} | Valid: {X_valid.shape} | Test: {X_test.shape}")
    
    # Initialize XGBoost Regressor
    logger.info("Initializing XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50, # limit overfitting on train set via validation check
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    logger.info("Training baseline model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100 # prints update every 100 trees to monitor progress
    )
    
    # Evaluate on test set and output basic metrics
    logger.info("Training complete. Evaluating on the held-out test set...")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logger.info(f"\nBaseline test metrics:")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R^2 : {r2:.4f}\n")
    
    # Save predictions and residuals to CSV for later analysis
    results_df = pd.DataFrame({
        'pIC50_true': y_test,
        'pIC50_pred': y_pred,
        'residual': y_test - y_pred,
    })
    results_out_dir = "data/results"
    os.makedirs(results_out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_out_dir, "xgboost_baseline_test_results.csv"), index=False)
    
    # Save metrics to JSON
    metrics = {"rmse": float(rmse), "r2": float(r2)}
    with open(os.path.join(results_out_dir, "xgboost_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info("Saved baseline test results to data/results/")
    
    # Save model
    os.makedirs(model_out_dir, exist_ok=True)
    model_path = os.path.join(model_out_dir, "xgb_baseline.json")
    model.save_model(model_path)
    logger.info(f"Saved baseline model to {model_path}")

if __name__ == "__main__":
    train_baseline()