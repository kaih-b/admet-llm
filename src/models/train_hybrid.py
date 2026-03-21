import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd
import json
from src.logger import get_console_logger
logger = get_console_logger(__name__)

def train_hybrid():
    embed_dir = "data/embeddings"
    model_out_dir = "models/hybrid"
    results_out_dir = "data/results"
    param_path = os.path.join(model_out_dir, "best_optuna_params.json")
    
    logger.info("Loading ChemBERTa embeddings...")
    X_train = np.load(os.path.join(embed_dir, "X_train_chemberta.npy"))
    y_train = np.load(os.path.join(embed_dir, "y_train.npy"))
    X_valid = np.load(os.path.join(embed_dir, "X_valid_chemberta.npy"))
    y_valid = np.load(os.path.join(embed_dir, "y_valid.npy"))
    X_test  = np.load(os.path.join(embed_dir, "X_test_chemberta.npy"))
    y_test  = np.load(os.path.join(embed_dir, "y_test.npy"))
    
    if os.path.exists(param_path):
        logger.info(f"Found optimized parameters at {param_path}. Loading...")
        with open(param_path, "r") as f:
            best_params = json.load(f)
    else:
        logger.warning("No optimized parameters found, falling back to default values.")
        best_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6
        }
    
    logger.info("Initializing XGBoost Regressor with optimized parameters:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")
    
    model = xgb.XGBRegressor(**best_params)
    
    logger.info("Training Hybrid model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100
    )
    
    logger.info("Evaluating on Test Set...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"\nHybird test metrics:")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R^2 : {r2:.4f}\n")
    
    os.makedirs(results_out_dir, exist_ok=True)
    
    # Save results
    data_dir = "data/processed"
    test_df = pd.read_csv(os.path.join(data_dir, "test_herg.csv"))
    results_df = pd.DataFrame({
        'smiles': test_df['smiles'],
        'pIC50_true': y_test,
        'pIC50_pred': y_pred,
        'resid': y_test - y_pred
    })
    results_path = os.path.join(results_out_dir, "hybrid_test_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Save metrics
    metrics_path = os.path.join(results_out_dir, "hybrid_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "rmse": float(rmse), 
            "r2": float(r2), 
            "best_params": best_params
        }, f, indent=4)
        
    # Save model
    os.makedirs(model_out_dir, exist_ok=True)
    model.save_model(os.path.join(model_out_dir, "xgb_hybrid.json"))
    
    logger.info("Saved optimized model, predictions, and metrics.")
if __name__ == "__main__":
    train_hybrid()