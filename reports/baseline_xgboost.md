# Baseline Model: XGBoost + Morgan Fingerprints

## Objective
Establish a machine learning baseline for hERG pIC50 prediction before training a computationally expensive LLM (ChemBERTa). 

## Procedure
* **Inputs:** 2D Morgan Fingerprints (Radius 2, 2048 bits) generated via RDKit `MorganGenerator`
* **Algorithm:** XGBoost Regressor (1000 estimators, early stopping)
* **Split:** 80/10/10 DeepChem Scaffold Split

## Test Set Results
* **RMSE:** 0.7399
* **R^2:** 0.2198

## Scientific Analysis
The $R^2$ of ~0.22 indicates that 2D molecular fingerprint-based models capture little of the variance in hERG binding affinity. 

### Why did it struggle?
The hERG channel can bind to many different molecule shapes. Because Morgan Fingerprints map basic 2D atom connections, they leave out the 3D geometry and electron properties necessary to make chemically accurate predictions.

### Next Steps 
Because 2D representations failed to capture the structural syntax required for this target, the next phase of this project will utilize a Transformer-based LLM (ChemBERTa), as was the initial plan, to learn the structural rules directly from the SMILES sequences.