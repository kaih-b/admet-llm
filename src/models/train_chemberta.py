import json
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
from src.logger import get_console_logger
logger = get_console_logger(__name__)

# Lock seeds for reproducibility
random_seed = 42
np.random.seed(42)
torch.manual_seed(42)

# Calcluates RMSE and R^2 during evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    r2 = r2_score(labels, predictions)
    return {"rmse": rmse, "r2": r2}

# Trains and tunes model
def tune_chemberta():

    # Check if the script is running inside Google Colab
    if 'google.colab' in sys.modules:
        logger.info("Google Colab cloud environment detected")
        base_path = "/content/drive/Othercomputers/My Mac/admet-llm"
    else:
        logger.info("Local environment detected")
        base_path = "."

    # Adjust pathnames to work for local and cloud environments
    data_dir = os.path.join(base_path, "data/processed")
    model_out_dir = os.path.join(base_path, "models/chemberta_herg")
    results_out_dir = os.path.join(base_path, "data/results")
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    
    # Load data
    logger.info("Loading train, valid, and test datasets...")
    train_df = pd.read_csv(os.path.join(data_dir, "train_herg.csv"))
    valid_df = pd.read_csv(os.path.join(data_dir, "valid_herg.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test_herg.csv"))

    # Convert pd dfs into HuggingFace Dataset objects
    hg_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "valid": Dataset.from_pandas(valid_df),
        "test":  Dataset.from_pandas(test_df)
    })
    
    # Auto-tokenize based on pretraining for model
    logger.info(f"Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure 512-character limit of ChemBERTa is not exceeded (safeguard, only affects very few molecules)
    def tokenize_function(examples):
        return tokenizer(
            examples["smiles"], 
            padding="max_length", 
            truncation=True, 
            max_length=512
            )
    logger.info("Tokenizing the datasets...")
    tokenized_datasets = hg_dataset.map(tokenize_function, batched=True)
    
    # Rename 'pIC50' to 'labels' (PyTorch looks for a column 'labels')
    tokenized_datasets = tokenized_datasets.rename_column("pIC50", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Load model
    logger.info("Loading pre-trained ChemBERTa model...")
    # num_labels tags as regression (not classification); critical here
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    # Setup training args
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        eval_strategy="epoch", # Check performance after each epoch
        save_strategy="epoch", # Save checkpoint after each epoch
        learning_rate=2e-5, # Very small; this model's pre-training should be good
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01, # Prevent overfitting
        load_best_model_at_end=True, # Automatically reload the best epoch (by validation RMSE)
        metric_for_best_model="rmse",
        greater_is_better=False, # RMSE --> lower is better
        report_to = "wandb",
        logging_steps=50,
        seed=random_seed
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics
    )
    
    # Train model
    logger.info("Starting tuning. This will take some time...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Training complete. Evaluating on the held-out Test Set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    logger.info(f"\nLLM (ChemBERTa) Test Metrics:")
    logger.info(f"RMSE: {test_results['eval_rmse']:.4f}")
    logger.info(f"R^2 : {test_results['eval_r2']:.4f}")
    
    # Save metrics to JSON
    metrics = {"rmse": float(test_results['eval_rmse']), "r2": float((test_results['eval_r2']))}
    with open(os.path.join(results_out_dir, "chemberta_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save final model and tokenizer
    trainer.save_model(model_out_dir)
    tokenizer.save_pretrained(model_out_dir)
    logger.info(f"Saved tuned LLM model to {model_out_dir}")
    
if __name__ == "__main__":
    tune_chemberta()