import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_trial_summary(trial_number, model_name, config, hyperparams, val_preds, val_targets, val_loss):
    trial_dir = f"trials/trial_{trial_number}_val"
    os.makedirs(trial_dir, exist_ok=True)
    summary_path = os.path.join(trial_dir, "trial_summary.txt")

    preds = np.array(val_preds)
    targets = np.array(val_targets)

    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    bias = np.mean(preds - targets)
    std_err = np.std(preds - targets)
    r2 = 1 - np.sum((preds - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)

    summary = f"""
📘 Trial Summary — Trial #{trial_number}
────────────────────────────────────────
🕹 Model Type      : {model_name}
🧠 Signal Type     : {config.get('signal_type', 'unknown')}
📊 Input FS        : {config.get('fs_ecg') if config.get('signal_type')=='ecg' else config.get('fs_ppg')}

🛠 Hyperparameters:
  • Batch Size     : {hyperparams['batch_size']}
  • Learning Rate  : {hyperparams['learning_rate']:.6f}
  • Weight Decay   : {hyperparams['weight_decay']:.6f}
  • Loss Function  : {config.get('loss_type', 'mse')}

📈 Validation Metrics:
  • Val Loss (Final): {val_loss:.4f}
  • MAE             : {mae:.4f}
  • MSE             : {mse:.4f}
  • RMSE            : {rmse:.4f}
  • Bias (Mean Err) : {bias:.4f}
  • Std Error       : {std_err:.4f}
  • R² Score        : {r2:.4f}

📅 Trial Folder     : {trial_dir}
────────────────────────────────────────
"""

    with open(summary_path, 'w') as f:
        f.write(summary.strip() + "\n")

    # Plot 1: Histogram of Prediction Errors
    plt.figure(figsize=(8, 4))
    sns.histplot(preds - targets, bins=30, kde=True)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Prediction Error (Prediction - Target)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(trial_dir, "error_histogram.png"))
    plt.close()

    # Plot 2: Scatter Plot of Prediction vs Target
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=targets, y=preds, alpha=0.6)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel("Ground Truth (Target)")
    plt.ylabel("Prediction")
    plt.title("Predicted vs Actual Respiratory Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(trial_dir, "scatter_pred_vs_true.png"))
    plt.close()

    return summary.strip()
