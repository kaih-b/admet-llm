import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
from src.logger import get_console_logger
logger = get_console_logger(__name__)

# Passes SMILES through ChemBERTa and extracts the [CLS] token embedding
def extract_embeddings(df, tokenizer, model, device, batch_size=32):
    model.eval()
    all_embeddings = []
    
    # Process in batches (avoid memory bottleneck)
    for i in tqdm(range(0, len(df), batch_size), desc="Extracting"):
        batch_smiles = df['smiles'].iloc[i:i+batch_size].tolist()
        
        # Cap length to BERT max
        inputs = tokenizer(batch_smiles, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Grab the last hidden state of the first [CLS] token
            # Shape: (batch_size, 768)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
            
    return np.vstack(all_embeddings)

def run_extraction():
    data_dir = "data/processed"
    embed_dir = "data/embeddings"
    os.makedirs(embed_dir, exist_ok=True)
    
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    logger.info(f"Loading {model_name} as a feature extractor...")
    
    # Load the pretrained model (without random head)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Configure to accept NVIDIA GPU or MPS or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using compute device: {device}")
    
    for split in ["train", "valid", "test"]:
        logger.info(f"Processing {split} set...")
        df = pd.read_csv(os.path.join(data_dir, f"{split}_herg.csv"))
        
        embeddings = extract_embeddings(df, tokenizer, model, device)
        y_values = df['pIC50'].values
        
        # Save X (embeddings) and y (targets)
        np.save(os.path.join(embed_dir, f"X_{split}_chemberta.npy"), embeddings)
        np.save(os.path.join(embed_dir, f"y_{split}.npy"), y_values)

if __name__ == "__main__":
    run_extraction()