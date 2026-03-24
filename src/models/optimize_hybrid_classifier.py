import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import json
from src.logger import get_console_logger
logger = get_console_logger(__name__)

def load_data():
    global X_train, y_train, X_valid, y_valid
    data_dir = "data/processed"
    embed_dir = "data/embeddings"

    THRESHOLD = 5.0
    logger.info(f"Loading data and binarizing labels (Threshold >= {THRESHOLD})...")

    # Load raw labels and binarize
    y_train_raw = pd.read_csv(os.path.join(data_dir, "train_herg.csv"))['pIC50'].values
    y_valid_raw = pd.read_csv(os.path.join(data_dir, "valid_herg.csv"))['pIC50'].values
    y_train = (y_train_raw >= THRESHOLD).astype(int)
    y_valid = (y_valid_raw >= THRESHOLD).astype(int)

    # Load ChemBERTa embeddings
    logger.info("Loading ChemBERTa embeddings...")
    X_train = np.load(os.path.join(embed_dir, "X_train_chemberta.npy"))
    X_valid = np.load(os.path.join(embed_dir, "X_valid_chemberta.npy"))

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
        # GPU acceleration
        "tree_method": "hist",
        "device": "cuda" if xgb.build_info().get('USE_CUDA', False) else "cpu",
        "early_stopping_rounds": 50,
        "eval_metric": "auc"
    }
    
    # Initialize classifier
    model = xgb.XGBClassifier(**param)
    
    # Train model with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False # Avoid terminal flooding
    )
    
    # Evaluate on the validation set using probability for AUC
    preds_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds_proba)
    
    return auc

def run_optimization():
    load_data()
    
    logger.info("Starting Optuna study for Classification...")
    # maximizing AUC
    study = optuna.create_study(direction="maximize", study_name="hybrid_classifier_optimization")
    study.optimize(objective, n_trials=100)
    
    logger.info("\nOptimization complete!")
    logger.info(f"Best Trial Number: {study.best_trial.number}")
    logger.info(f"Best Validation ROC-AUC: {study.best_value:.4f}")
    logger.info("Best Parameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")
    
    # Save optimized model parameters
    out_dir = "models/hybrid_classifier"
    os.makedirs(out_dir, exist_ok=True)
    param_path = os.path.join(out_dir, "best_optuna_params_classifier.json")
    
    with open(param_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    logger.info(f"Saved optimized hyperparameters to {param_path}")
    
if __name__ == "__main__":
    run_optimization()