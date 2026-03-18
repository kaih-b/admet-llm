# Baseline Models: Graph Neural Networks (GCN & GAT)

## Objective
Establish deep learning baselines using 2D graph representations. This includes both a standard Graph Convolutional Network (GCN) and a Graph Attention Network (GAT) to test the efficacy of attention mechanisms on molecular graphs before moving to sequence-based transformers.

## Procedure
* **Inputs**: Molecular Graphs (Nodes = Atoms, Edges = Bonds) generated via DeepChem `MolGraphConvFeaturizer`
* **Algorithms**:
  * **GCN**: PyTorch GCN (2 graph convolution layers, early stopping with `patience = 10`)
  * **GAT**: PyTorch GAT (8 attention heads, early stopping with `patience = 10`)
* **Split**: 80/10/10 DeepChem Scaffold Split

## Test Set Results
* **GCN**:
  * **RMSE:** 0.8115
  * **R^2:** 0.0614
* **GAT**:
  * **RMSE:** 0.9269
  * **R^2:** 0.2245

## Scientific Analysis
This baseline study revealed two key insights into molecular representation for the hERG channel:

1. **The Necessity of Attention**: The GCN completely failed to model the variance ($R^2 ~ 0.06$). Simply averaging the features of neighboring atoms (standard convolution) essentially nullifies the chemical signals that drive toxicity. Adding multi-head attention (GAT) allowed the network to learn which specific bonds and atoms to focus on, causing the $R^2$ to jump to ~0.22. 

2. **The 2D Ceiling**: The GAT's $R^2$ of 0.2245 essentially tied the XGBoost baseline (0.2198). However, the GAT suffered a higher RMSE (0.92 vs 0.74), indicating that while it captured the overall variance trend better than the GCN, it made larger errors on outlier molecules. 

### Why did they struggle?
While the GAT successfully captured topological connectivity, it still lacks 3D spatial coordinates. Because the hERG channel accommodates a wide variety of shapes, binding affinity is heavily driven by 3D flexibility and folding geometry. A 2D topological graph - even one with attention - cannot accurately represent these 3D dynamics.

### Next Steps 
The progression from isolated 1D vectors (XGBoost) to interconnected 2D graphs (GCN, GAT) has established a firm mathematical ceiling of roughly 22% explained variance. The next phase will shift to a sequence-based approach. By utilizing a transformer-based LLM (ChemBERTa), the model's self-attention mechanism will be tested to see if it can infer 3D structural rules directly directly from SMILES sequences.