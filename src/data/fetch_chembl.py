# Purpose: fetches hERG (toxicity target) IC50 (substance potency) bioactivity data from ChEMBL database
import pandas as pd
from chembl_webresource_client.new_client import new_client
import os

def fetch_herg_data(output_path: str):
    print("Connecting to ChEMBL API...")
    activity = new_client.activity
    
    # Filter for hERG target (HEMBL240), IC50 metric, and exact measurements (i.e. not bounds)
    query = activity.filter(target_chembl_id="CHEMBL240", standard_type="IC50", standard_relation="=")
    print(f"Retrieving {len(query)} records...")
    data = list(query)
    
    # Convert to a df and filter for essential columns, dropping NaNs for SMILES and IC50 (standard_value)
    df = pd.DataFrame(data)
    cols_to_keep = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units']
    df = df[cols_to_keep].dropna(subset=['canonical_smiles', 'standard_value'])
    
    # Convert IC50 standard_value to numeric, drop errors
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna(subset=['standard_value'])
    
    # Save to raw data folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved raw data to {output_path}")

if __name__ == "__main__":
    fetch_herg_data("../../data/raw/herg_raw.csv")