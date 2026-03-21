# Large Language Model: ChemBERTa (Transformer)

## Objective
Determine if a pre-trained, sequence-based Large Language Model (ChemBERTa) can infer 3D structural rules and electronic properties directly from SMILES strings, aiming to break the performance ceiling established by the XGBoost and GNN baselines.

## Procedure
* **Inputs**: 1D SMILES sequences (Tokenized, `max_length = 512`).
* **Algorithm**: Fine-tuned `DeepChem/ChemBERTa-77M-MTR` (77 million parameters).
  * **Architecture Setup**: Base classification head replaced with a regression head (`num_labels=1`, MSE loss).
  * **Hyperparameters**: 5 epochs, Batch Size = 16, Learning Rate = 2e-5, Weight Decay = 0.01.
  * **Early Stopping**: Automated best-model reloading based on Validation RMSE.
* **Split**: 80/10/10 DeepChem Scaffold Split.

## Test Set Results
* **RMSE**: 0.8641
* **R^2**: -0.0640

## Scientific Analysis
These results provide a definitive benchmark on the limitations of applying large representation-learning architectures to smaller, specialized chemical datasets. 

1. **Overfitting Behavior**: The negative $R^2$ score (-0.064) indicates that the model performed worse than a simplistic mean-prediction baseline on the held-out test set. Though the 77-million parameter network possessed enough capacity to quickly minimize error on the training set, it failed to generalize its learning to the test set, indicating severe overfitting. 

2. **Data Starvation**: The model successfully learned the basic rules of the ~7,600 training molecules, but lacked the sample volume required to infer universal 3D binding mechanics without memorizing the training data.

### Visual Diagnostics

**1. Training vs. Validation Loss**
![Training vs Validation Loss](../assets/wandb_loss_curve.png)
The loss exhibits late-stage overfitting. The validation loss (orange) hits a hard generalization floor early in the run. By the final steps, the validation error spikes slightly upward while the training error (blue) remains low.

**2. Residuals vs. Predicted & Error Distribution**
![Residual Analysis](../assets/chemberta_residuals.png)
* **Variance Compression (Left)**: In a proper model, residuals are randomly scattered around the zero-error line. Here, the model compresses a vast majority of its predictions into a narrow band between 4.0 and 6.0. It failed to identify extreme values, resulting in a distinct diagonal bias where it systematically under-predicts strong blockers and over-predicts weak ones.
* **Under-Prediction Skew (Right)**: The error distribution is right-skewed. The long tail of positive residuals indicates that the LLM was particularly poor at identifying the most dangerous hERG blockers, often guessing near the dataset mean instead of recognizing highly toxic 3D indicators.

### Why did it struggle?
While transformers excel at language translation and broad chemical property prediction (where pre-training data is vast), hERG binding is an distinct 3D phenomenon. A 1D text representation (SMILES), even when processed by a massive self-attention network, cannot effectively map what is required to block the hERG channel without orders of magnitude more training data.

### Next Steps
The progression from 1D mathematical arrays (XGBoost: $R^2$ ~0.22) to 2D topological graphs (GCN/GAT: $R^2$ ~0.06 to -0.22) and finally to 1D sequence-based LLMs (ChemBERTa: $R^2$ -0.06) yields the following conclusion: **For specific, 3D-dependent targets (like hERG) with small datasets (< 10,000 samples), gradient boosting algorithms paired with 2D representations (Morgan Fingerprints) outperform complex neural architectures.** The final phase of this project will focus on consolidating these benchmarking results by utilizing the hybrid approach, using ChemBERTa for embeddings (purely a feature extractor) and XGBoost predictions.