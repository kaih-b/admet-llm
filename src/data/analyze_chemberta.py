import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.logger import get_console_logger
logger = get_console_logger(__name__)

def analyze_residuals():
    # Define paths
    results_dir = "data/results"
    plots_dir = "assets"
    os.makedirs(plots_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "chemberta_test_predictions.csv")

    # Load the predictions
    logger.info(f"Loading predictions from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Calculate residuals
    df["residual"] = df["pIC50"] - df["pIC50_pred"]
    df["absolute_error"] = abs(df["residual"])

    # Identify biggest failures
    logger.info("\nTop 5 Worst Predictions:")
    worst_df = df.sort_values(by="absolute_error", ascending=False).head(5)
    for index, row in worst_df.iterrows():
        logger.info(f"SMILES: {row['smiles']}")
        logger.info(f"True pIC50: {row['pIC50']:.2f} | Predicted: {row['pIC50_pred']:.2f} | Residual: {row['residual']:.2f}\n")

    # Generate plots
    logger.info("Generating residual analysis plots...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Residuals vs. Predicted
    sns.scatterplot(
        x="pIC50_pred", 
        y="residual", 
        data=df, 
        alpha=0.6, 
        edgecolor="k",
        color="#4C72B0",
        ax=axes[0]
    )
    # Zero-error baseline for reference
    axes[0].axhline(0, color='red', linestyle='--', lw=2, label="Zero Error")
    axes[0].set_title("Residuals vs. Predicted pIC50", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted pIC50 (ChemBERTa)", fontsize=12)
    axes[0].set_ylabel("Residual (True - Predicted)", fontsize=12)
    axes[0].legend()

    # Plot 2: Residual Distribution
    sns.histplot(
        df["residual"], 
        kde=True, 
        color="#55A868", 
        edgecolor="k",
        ax=axes[1]
    )
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set_title("Distribution of Residual Errors", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Residual Value", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)

    # Format and save
    plt.suptitle("ChemBERTa Test Set: Residual Analysis", fontsize=16, y=1.05, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "chemberta_residuals.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved residual analysis to {plot_path}")
    plt.close()

if __name__ == "__main__":
    analyze_residuals()