import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import get_console_logger
logger = get_console_logger(__name__)

def main():
    data_dir = "data/processed"
    embeddings_dir = "data/embeddings"
    results_dir = "data/results"
    model_out_dir = "models/hybrid_classifier"
    param_path = os.path.join(model_out_dir, "best_optuna_params_classifier.json")
    os.makedirs(model_out_dir, exist_ok=True)

    # Set Toxicity threshold: pIC50 > 5.0 is considered toxic, < 5.0 is considered only semi-toxic or non-toxic
    # This classifier immediately rules out toxic molecules; all that remain are not necessarily non-toxic
    THRESHOLD = 5.0
    logger.info(f"Binarizing labels with pIC50 Threshold: {THRESHOLD} (>= {THRESHOLD} is Toxic)")

    # Load the raw dfs to extract pIC50
    logger.info("Loading original CSVs for labels...")
    y_train_raw = pd.read_csv(os.path.join(data_dir, "train_herg.csv"))['pIC50'].values
    y_valid_raw = pd.read_csv(os.path.join(data_dir, "valid_herg.csv"))['pIC50'].values
    y_test_raw = pd.read_csv(os.path.join(data_dir, "test_herg.csv"))['pIC50'].values

    # Binarize labels
    y_train = (y_train_raw >= THRESHOLD).astype(int)
    y_valid = (y_valid_raw >= THRESHOLD).astype(int)
    y_test = (y_test_raw >= THRESHOLD).astype(int)

    # Load ChemBERTa embeddings (same as train_hybrid.py)
    logger.info("Loading ChemBERTa embeddings...")
    X_train = np.load(os.path.join(embeddings_dir, "X_train_chemberta.npy"))
    X_valid = np.load(os.path.join(embeddings_dir, "X_valid_chemberta.npy"))
    X_test = np.load(os.path.join(embeddings_dir, "X_test_chemberta.npy"))

    # Load Optimized Parameters
    with open(param_path, "r") as f:
        best_params = json.load(f)

    logger.info("Initializing XGBoost Classifier with optimized parameters:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")
    
    # Add non-searchable parameters back in
    best_params['random_state'] = 42
    best_params['eval_metric'] = "auc"
    best_params['early_stopping_rounds'] = 50
    best_params['tree_method'] = "hist"
    best_params['device'] = "cuda" if xgb.build_info().get('USE_CUDA', False) else "cpu"
    
    # Initialize optimizer classifier
    xgb_classifier = xgb.XGBClassifier(**best_params)

    # Train the model, using validation for early stopping
    xgb_classifier.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100
    )

    # Evaluate on the Test Set
    logger.info("Evaluating on the Test Set...")
    
    # Predict absolute classes (0 or 1)
    y_pred_class = xgb_classifier.predict(X_test)
    
    # Predict probabilities (used for ROC-AUC)
    y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1] 

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)
    auc = roc_auc_score(y_test, y_pred_proba)

    logger.info("\nHybrid Classifier Test Metrics:")
    logger.info(f"Accuracy : {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f} (when predicts toxic, how often is it right?)")
    logger.info(f"Recall   : {recall:.4f} (out of all toxic, how many were caught?)")
    logger.info(f"F1-Score : {f1:.4f}")
    logger.info(f"ROC-AUC  : {auc:.4f} (ability to distinguish toxicity)")

    # Save model
    save_path = os.path.join(model_out_dir, "xgb_classifier_hybrid.json")
    xgb_classifier.save_model(save_path)
    logger.info(f"Model saved to {save_path}")
    
    # Save metrics to JSON
    metrics_dict = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1-score": float(f1),
        "ROC-AUC": float(auc)
    }
    metrics_path = os.path.join(results_dir, "hybrid_classifier_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")

    # Predictions
    results_df = pd.DataFrame({
        "true": y_test,
        "predicted": y_pred_class,
        "predicted_probability": y_pred_proba
    })
    predictions_path = os.path.join(results_dir, "hybrid_classifier_test_results.csv")
    results_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved test predictions to {predictions_path}")

    # Save model
    save_path = os.path.join(model_out_dir, "xgb_classifier_hybrid.json")
    xgb_classifier.save_model(save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()