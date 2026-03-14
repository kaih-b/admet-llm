# Purpose: curate and scaffold split raw ChEMBL entries
# Scaffold split groups molecules by structure
import pandas as pd
import numpy as np
from rdkit import Chem
from deepchem.splits import ScaffoldSplitter
import deepchem as dc
import os
import logging

# Configure console-only logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("curate_data")

# Normalizes (canonicalizes) SMILES strings using RDKit
def normalize_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except:
        return None

def process_and_split(input_path: str, output_dir: str):
    logger.info(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info("Canonicalizing SMILES using RDKit...")
    df['canonical_smiles'] = df['canonical_smiles'].apply(normalize_smiles)
    df = df.dropna(subset=['canonical_smiles'])
    
    # Aggregate medians for duplicates after normalizing
    logger.info("Aggregating duplicates (taking the median IC50)...")
    df = df.groupby('canonical_smiles').agg({
        'standard_value': 'median',
        'molecule_chembl_id': 'first'
    }).reset_index()
    
    # Convert IC50 (nM) to pIC50 (negative log molar) for ML scaling
    # pIC50 = -log10(IC50 * 10^-9) = 9 - log10(IC50), add epsilon to avoid zero errors
    logger.info("Calculating pIC50 values...")
    df['pIC50'] = 9 - np.log10(df['standard_value'] + 1e-10)
    
    # 80/10/10 scaffoled split
    logger.info("Performing Scaffold Split via DeepChem...")
    dataset = dc.data.DiskDataset.from_numpy(
        X=df['canonical_smiles'].values, 
        y=df['pIC50'].values,
        ids=df['canonical_smiles'].values)
    splitter = ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    train_df = pd.DataFrame({'smiles': train_dataset.ids, 'pIC50': train_dataset.y.flatten()})
    valid_df = pd.DataFrame({'smiles': valid_dataset.ids, 'pIC50': valid_dataset.y.flatten()})
    test_df = pd.DataFrame({'smiles': test_dataset.ids, 'pIC50': test_dataset.y.flatten()})
    
    # Save train and test sets
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_herg.csv"), index=False)
    valid_df.to_csv(os.path.join(output_dir, "valid_herg.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_herg.csv"), index=False)
    logger.info(f"Split complete. Train size: {len(train_df)} | Valid: {len(valid_df)} | Test size: {len(test_df)}")
    logger.info(f"Saved processed data to {output_dir}")
    
if __name__ == "__main__":
    process_and_split("data/raw/herg_raw.csv", "data/processed/")