import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd
import json
from src.logger import get_console_logger
logger = get_console_logger(__name__)

def load_data():
    global X_train, y_train, X_valid, y_valid
    embed_dir = "data/embeddings"

    logger.info("Loading embeddings...")
    X_train = np.load(os.path.join(embed_dir, "X_train_chemberta.npy"))
    y_train = np.load(os.path.join(embed_dir, "y_train.npy"))
    X_valid = np.load(os.path.join(embed_dir, "X_valid_chemberta.npy"))
    y_valid = np.load(os.path.join(embed_dir, "y_valid.npy"))

def objective(trial):
    # Define hyperparameter search space
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        # GPU acceleration (Google Colab T4 compat)
        "tree_method": "hist",
        "device": "cuda",
        "early_stopping_rounds": 50
    }
    
    # Initialize model
    model = xgb.XGBRegressor(**param)
    
    # Train model with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False # Avoid terminal flooding
    )
    
    # Evaluate on the validation set
    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    
    return rmse

def run_optimization():
    load_data()
    
    logger.info("Starting Optuna study...")
    study = optuna.create_study(direction="minimize", study_name="hybrid_xgboost_optimization")
    study.optimize(objective, n_trials=100)
    
    logger.info("\nOptimization complete!")
    logger.info(f"Best Trial Number: {study.best_trial.number}")
    logger.info(f"Best Validation RMSE: {study.best_value:.4f}")
    logger.info("Best Parameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")
    
    # Save optimized model for easy hyperparam initialization
    os.makedirs("models/hybrid", exist_ok=True)
    param_path = "models/hybrid/best_optuna_params.json"
    with open(param_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    logger.info(f"Saved optimized hyperparameters to {param_path}")
    
if __name__ == "__main__":
    run_optimization()