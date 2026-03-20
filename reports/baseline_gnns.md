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
  * **RMSE**: 0.8115
  * **R^2**: 0.0614
* **GAT**:
  * **RMSE**: 0.9269
  * **R^2**: -0.2245

## Scientific Analysis
This baseline study revealed two key insights into molecular representation and model complexity for the hERG channel:

1. **Complexity & Overfitting**: Unlike XGBoost, which successfully captured some variance ($R^2$ ~0.23), training graph neural networks from scratch on this dataset largely failed. The simple GCN barely learned anything beyond the mean ($R^2$ ~ 0.06). However, adding complex multi-head attention (GAT) actually worsened performance to the point of a negative $R^2$ (-0.22). This indicates that the high-parameter GAT overfit the training data and lost the ability to generalize to test scaffolds.

2. **Deep Learning Instability**: The tuned XGBoost model (RMSE 0.74) proved to be far more robust to the noisy nature of the hERG dataset than deep learning models. The GAT's high RMSE (0.92) shows that when it guessed wrong, it missed by massive margins.

### Why did they struggle?
While GNNs successfully capture topological connectivity, training them from a random initialization requires massive amounts of data. Furthermore, they still lack explicit 3D spatial coordinates. Because the hERG channel accommodates a wide variety of shapes, binding affinity is heavily driven by 3D flexibility and folding geometry. A 2D topological graph - especially one trained from scratch on a limited dataset - cannot accurately represent these dynamics without overfitting.

### Next Steps 
The progression from isolated 1D vectors (XGBoost) to interconnected 2D graphs (GCN, GAT) has established a firm mathematical ceiling of roughly 22% explained variance. The next phase will shift to a sequence-based approach. By utilizing a transformer-based LLM (ChemBERTa), the model's self-attention mechanism will be tested to see if it can infer 3D structural rules directly directly from SMILES sequences.