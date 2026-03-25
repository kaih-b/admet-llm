import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.logger import get_console_logger
logger = get_console_logger(__name__)

def main():
    results_dir = "data/results"
    figs_dir = "assets/"
    predictions_path = os.path.join(results_dir, "hybrid_classifier_test_results.csv")
    
    if not os.path.exists(predictions_path):
        logger.error(f"Could not find {predictions_path}. Run the classifier test script first.")
        return

    logger.info("Loading prediction data...")
    df = pd.read_csv(predictions_path)
    
    y_true = df['true']
    y_pred_class = df['predicted']
    y_pred_proba = df['predicted_probability']

    # Set universal plotting style
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # Confusion matrix
    logger.info("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred_class)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Safe (0)', 'Toxic (1)'],
                yticklabels=['Safe (0)', 'Toxic (1)'])
    
    plt.title('hERG Toxicity Confusion Matrix\n(Threshold: pIC50 $\geq$ 5.0)', pad=15, fontweight='bold')
    plt.ylabel('True Assay Label', fontweight='bold')
    plt.xlabel('Hybrid Prediction', fontweight='bold')
    
    cm_path = os.path.join(figs_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {cm_path}")

    # ROC Curve
    logger.info("Generating ROC Curve...")
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#d95f02', lw=2.5, label=f'Hybrid Model (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Guess')
    
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', pad=15, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, shadow=True)
    
    roc_path = os.path.join(figs_dir, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {roc_path}")

if __name__ == "__main__":
    main()