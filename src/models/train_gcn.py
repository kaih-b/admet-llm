import os
import logging
from rdkit import RDLogger
import warnings

# Mute errors to avoid terminal flooding
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("deepchem").setLevel(logging.ERROR)
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=UserWarning)

# Apple MPS and DGN compat
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import json
import pandas as pd
import numpy as np
import deepchem as dc
from sklearn.metrics import mean_squared_error, r2_score
import torch
from src.logger import get_console_logger
logger = get_console_logger(__name__)

# Lock seeds for reproducibility
random_seed = 42
np.random.seed(42)
torch.manual_seed(42)

# Load the SMILES CSV and convert each into a Graph ConvMol obect
def load_and_featurize_graphs(file_path: str, featurizer: dc.feat.Featurizer):
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert SMILES to Graph objects
    features = featurizer.featurize(df['smiles'].values)
    
    # Filter 'None' values to prevent the neural net from crashing
    valid_indices = [i for i, f in enumerate(features) if f is not None]
    
    X_valid = features[valid_indices]
    y_valid = df['pIC50'].values[valid_indices].reshape(-1, 1)
    smiles_valid = df['smiles'].values[valid_indices]
    
    # Convert to DeepChem Dataset object
    dataset = dc.data.NumpyDataset(X=X_valid, y=y_valid, ids=smiles_valid)
    return dataset, df.iloc[valid_indices]

def train_gnn():
    data_dir = "data/processed"
    model_out_dir = "models/baseline"
    results_out_dir = "data/results"
    
    # Initialize the featurizer (edges include bond feature)
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    
    # Load data
    train_dataset, _ = load_and_featurize_graphs(os.path.join(data_dir, "train_herg.csv"), featurizer)
    valid_dataset, _ = load_and_featurize_graphs(os.path.join(data_dir, "valid_herg.csv"), featurizer)
    test_dataset, test_df = load_and_featurize_graphs(os.path.join(data_dir, "test_herg.csv"), featurizer)
    logger.info(f"Graph datasets created. Train: {len(train_dataset)} | Valid: {len(valid_dataset)} | Test: {len(test_dataset)}")
    
    # Initialize GCN model (baseline NN variables, tuning could yield better results)
    logger.info("Initializing PyTorch Graph Convolutional Network (GCN)...")
    model = dc.models.GCNModel(
        n_tasks=1,
        mode='regression',
        graph_conv_layers=[64, 64],
        dropout=0.2,
        batch_size=64,
        learning_rate=0.001,
        model_dir=model_out_dir,
        random_seed=random_seed,
        device=torch.device('cpu') # Apple MPS and DGN compat
    )

    # Configure early stopping
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    best_val_r2 = -np.inf
    patience = 10
    patience_counter = 0
    best_epoch = 0

    logger.info("Training GCN with early stopping (patience=10)...")
    for epoch in range(100):
        model.fit(train_dataset, nb_epoch=1)
        scores = model.evaluate(valid_dataset, [metric])
        val_r2 = scores['pearson_r2_score']
        logger.info(f"Epoch {epoch+1:3d} | Val R^2: {val_r2:.4f} | Best: {best_val_r2:.4f} | Patience: {patience_counter}/{patience}")

        if val_r2 > best_val_r2 + 1e-4: # threshold to avoid noise
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            patience_counter = 0
            model.save_checkpoint() # save best weights
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered. Best epoch: {best_epoch} | Best Val R^2: {best_val_r2:.4f}")
                break

    # Restore best weights before evaluating on test set
    model.restore()
    
    # Evaluate on the  Test Set
    logger.info("Training complete. Evaluating on the held-out test set...")
    
    # Flatten prediction arrays and output test metrics
    y_pred = model.predict(test_dataset).flatten()
    y_true = test_dataset.y.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    logger.info(f"\nGCN Test Metrics:")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R^2 : {r2:.4f}")
    
    # Save predictions and residuals to CSV for later analysis
    os.makedirs(results_out_dir, exist_ok=True)
    results_df = pd.DataFrame({
        'smiles': test_dataset.ids,
        'pIC50_true': y_true,
        'pIC50_pred': y_pred,
        'residual': y_true - y_pred
    })
    results_df.to_csv(os.path.join(results_out_dir, "gcn_baseline_test_results.csv"), index=False)
    
    # Save metrics to JSON
    metrics = {"rmse": float(rmse), "r2": float(r2)}
    with open(os.path.join(results_out_dir, "gcn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info("Saved baseline test results to data/results/")
    logger.info(f"Saved GCN model to {model_out_dir}")

if __name__ == "__main__":
    train_gnn()