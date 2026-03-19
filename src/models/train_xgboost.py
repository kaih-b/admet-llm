# Purpose: establish regression performance baseline using Morgan Fingerprints to later determine necessity of LLM use (pricier, computationally expensive)
import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
import optuna
from src.logger import get_console_logger
logger = get_console_logger(__name__)

# Lock random seeds for reproducibility)
random_seed = 42
np.random.seed(42)

# HELPER FUNCTIONS:
# Convert a SMILES string to a Morgan fingerprint array in np
def generate_fingerprints(smiles_list, radius=2, nBits=2048):
    mfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    features = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = mfp_gen.GetFingerprintAsNumPy(mol)
            features.append(fp)
            valid_indices.append(i)
            
    return np.array(features), valid_indices

# Load a CSV into a dataframe
def load_and_featurize(file_path: str):
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    logger.info("Generating Morgan Fingerprints...")
    # Stack the individual arrays into a 2D matrix
    X, valid_indices = generate_fingerprints(df['smiles'].values)
    X = np.vstack(X)
    y = df['pIC50'].values[valid_indices]
    return X, y, df.iloc[valid_indices]

# MAIN FUNCTION:
def train_tune_xgboost():
    data_dir = "data/processed"
    model_out_dir = "models/baseline"
    results_out_dir = "data/results"
    
    # Load data
    X_train, y_train, _ = load_and_featurize(os.path.join(data_dir, "train_herg.csv"))
    X_valid, y_valid, _ = load_and_featurize(os.path.join(data_dir, "valid_herg.csv"))
    X_test, y_test, test_df = load_and_featurize(os.path.join(data_dir, "test_herg.csv"))
    logger.info(f"Feature matrix shapes - Train: {X_train.shape} | Valid: {X_valid.shape} | Test: {X_test.shape}")
    
    # Optuna objective function
    def objective(trial):
        # Hyperparams search space
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'random_state': random_seed,
            'n_jobs': -1
        }
        model = xgb.XGBRegressor(**param, early_stopping_rounds = 20)
        
        # Early stopping to prevent overfitting
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        
        # Evaluate on valdiation set
        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        
        return rmse

    # Run the Optuna optimizer
    logger.info("Starting Optuna hyperparameter tuning (50 Trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    logger.info(f"Tuning complete! Best validation RMSE: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Train the optimized model using the best parameters
    logger.info("Training final XGBoost model with optimized parameters...")
    final_model = xgb.XGBRegressor(**best_params, random_state=random_seed, n_jobs=-1, early_stopping_rounds=50)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    
    logger.info("Evaluating optimized model on the held-out Test Set...")
    y_pred = final_model.predict(X_test)
    
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_r2 = r2_score(y_test, y_pred)
    logger.info(f"Optimized XGBoost Test Metrics:")
    logger.info(f"RMSE: {final_rmse:.4f}")
    logger.info(f"R^2 : {final_r2:.4f}")
    
    # Save results
    os.makedirs(results_out_dir, exist_ok=True)
    results_df = pd.DataFrame({
        'smiles': test_df['smiles'],
        'pIC50_true': y_test,
        'pIC50_pred': y_pred,
        'residual': y_test - y_pred
    })
    results_df.to_csv(os.path.join(results_out_dir, "xgboost_baseline_test_results.csv"), index=False)
    
    # Save model params
    os.makedirs(model_out_dir, exist_ok=True)
    final_model.save_model(os.path.join(model_out_dir, "xgboost_baseline.json"))
    
    # Save metrics
    with open(os.path.join(results_out_dir, "xgboost_metrics.json"), "w") as f:
        json.dump({"rmse": float(final_rmse), "r2": float(final_r2), "best_params": best_params}, f, indent=4)

    logger.info("Saved optimized model, predictions, and metrics.")

if __name__ == "__main__":
    train_tune_xgboost()