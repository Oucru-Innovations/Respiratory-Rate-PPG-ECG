import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import logging
import mlflow
import optuna
import colorlog
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split

from wearable_data import WearableDataset
from model_pool import ViTRegressor, CNNLSTMRegressor, \
    LiteTransformerRegressor, WaveNetRegressor, LiteCNNRegressor, \
    ConvNeXtRegressor, WaveViTRegressor, LiteViTRegressor, \
    ViTWaveRegressor
from model_utils import loss_uniform_spread_l2, mse_loss, mae_loss, huber_loss, smooth_l1_loss
from report import save_trial_summary
# Setup environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============ Logging ============ #
def setup_logging():
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s | %(levelname)s | %(message)s',
        log_colors={
            'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
            'ERROR': 'red', 'CRITICAL': 'bold_red',
        }
    ))
    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logging()

# ============ Config ============ #
CONFIG_PATH = 'dl/config/config.json'
CONFIG_TUNING_PATH = 'dl/config/config_tuning.json'

def load_config(path=CONFIG_PATH):
    with open(path, 'r') as f:
        return json.load(f)

def load_tuning_config(path=CONFIG_TUNING_PATH):
    with open(path, 'r') as f:
        return json.load(f)

# ============ Data Loading ============ #
def read_csv_data(path, label, is_target=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at {path}")
    data = np.loadtxt(path, delimiter=',')
    return data.reshape(-1, 1) if is_target else data

def load_datasets(split=0.2, plot_samples=False):
    data = read_csv_data('dataset/train_data.csv', "Train Data")
    target = read_csv_data('dataset/train_target.csv', "Train Target", is_target=True)
    train_data, val_data, train_target, val_target = train_test_split(
        data, target, test_size=split, shuffle=True, random_state=42
    )
    return (
        WearableDataset(train_data, train_target, plot_samples=plot_samples),
        WearableDataset(val_data, val_target)
    )

# ============ Loss Factory ============ #
def get_loss_function(cfg):
    name = cfg.get("loss_type", "mse").lower()
    return {
        "mse": mse_loss(),
        "mae": mae_loss(),
        "huber": huber_loss(cfg.get("huber_delta", 1.0)),
        "smooth_l1": smooth_l1_loss(),
        "uniform_spread_l2": loss_uniform_spread_l2()
    }.get(name, mse_loss())

# ============ Model Factory ============ #
def get_model(cfg):
    model_map = {
        "vit": ViTRegressor,
        "cnnlstm": CNNLSTMRegressor,
        "lite_transformer": LiteTransformerRegressor,
        "wavenet": WaveNetRegressor,
        "lite_cnn": LiteCNNRegressor,
        "convnext": ConvNeXtRegressor,  # best one
        "wavevit": WaveViTRegressor,
        "lite_vit": LiteViTRegressor,
        "vit_wave": ViTWaveRegressor
    }
    model_class = model_map.get(cfg.get("model_type", "vit").lower(), ViTRegressor)
    fs = cfg["fs_ecg"] if cfg["signal_type"] == "ecg" else cfg["fs_ppg"]
    return model_class(fs=fs)

# ============ LR Scheduler ============ #
def cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def _lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))))
    return LambdaLR(optimizer, _lr_lambda)

# ============ Training ============ #
def train_model(config, hyperparams, trial=None):
    try:
        logger.info("⏳ Loading Datasets...")
        test_data = read_csv_data('dataset/test_data.csv', "Test Data")
        test_targets = read_csv_data('dataset/test_target.csv', "Test Target", is_target=True)
        test_dataset = WearableDataset(test_data, test_targets)

        train_dataset, val_dataset = load_datasets(config.get("validation_split", 0.2), config.get("plot_samples", False))
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(config).to(device)
        criterion = get_loss_function(config)
        optimizer = Adam(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
        scheduler = cosine_warmup_scheduler(optimizer, warmup_epochs=5, total_epochs=config["epochs"])

        best_val_loss = float('inf')
        patience_counter = 0

        trial_number = trial.number if trial else 'manual'
        trial_dir = f"trials/trial_{trial_number}_val"
        os.makedirs(trial_dir, exist_ok=True)

        mlflow.start_run()
        mlflow.log_params({**config, **hyperparams})

        for epoch in tqdm(range(1, config["epochs"] + 1), desc="Epochs"):
            model.train()
            train_loss = 0.0
            for original_input, inputs, targets in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
                inputs, targets = inputs.to(device), targets.to(device).squeeze()
                optimizer.zero_grad()
                preds = model((original_input.to(device), inputs)).squeeze() # ensure remains shape as targets
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

            # Validation
            model.eval()
            val_loss, val_preds, val_true = 0.0, [], []
            with torch.no_grad():
                for original_input, inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device).squeeze()
                    preds = model((original_input.to(device), inputs)).squeeze() # ensure remains shape as targets
                    loss = criterion(preds, targets)
                    val_loss += loss.item() * inputs.size(0)
                    val_preds.extend(preds.cpu().numpy().flatten())
                    val_true.extend(targets.cpu().numpy().flatten())

            val_loss /= len(val_loader.dataset)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")

            # Save predictions
            pred_path = os.path.join(trial_dir, f"val_predictions_epoch_{epoch}.csv")
            pd.DataFrame({'Target': val_true, 'Prediction': val_preds}).to_csv(pred_path, index=False)
            mlflow.log_artifact(pred_path)

            if val_loss + config["early_stopping"]["min_delta"] < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pth")
                logger.info("🔖 New best model saved.")

                # ⏳ Save predictions for summary
                val_predictions_all = val_preds.copy()
                val_targets_all = val_true.copy()
            else:
                patience_counter += 1
                logger.info(f"⏳ No improvement. Early stopping: {patience_counter}/{config['early_stopping']['patience']}")
                if patience_counter >= config["early_stopping"]["patience"]:
                    logger.warning("⛔ Early stopping triggered.")
                    break


            if trial:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    logger.warning("Trial pruned by Optuna.")
                    raise optuna.exceptions.TrialPruned()
            trial_dir = f"trials/trial_{trial.number}_val"
            os.makedirs(trial_dir, exist_ok=True)

            csv_path = os.path.join(trial_dir, f"val_predictions_epoch_{epoch}.csv")
            val_df = pd.DataFrame({'Target': val_targets_all, 'Prediction': val_predictions_all})
            val_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)

            
            scheduler.step()

        mlflow.log_artifact('best_model.pth')

        # 🔍 Save trial summary
        # Save summary at best epoch
        if val_loss == best_val_loss:
            summary_text = save_trial_summary(
                trial_number=trial.number,
                model_name=config.get("model_type", "vit"),
                config=config,
                hyperparams=hyperparams,
                val_preds=val_predictions_all,
                val_targets=val_targets_all,
                val_loss=val_loss
            )
            logger.info("\n" + summary_text)

        mlflow.end_run()
        return best_val_loss
    except Exception as e:
        logger.exception(f"Error in training: {e}")
        mlflow.end_run()
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            if trial:
                trial.set_user_attr("OOM", True)
                raise optuna.exceptions.TrialPruned()
        raise

# ============ Tuning ============ #
def objective(trial):
    cfg = load_config()
    tuning_config = load_tuning_config()
    hyperparams = {
        'batch_size': trial.suggest_categorical('batch_size', tuning_config["batch_size"]),
        'learning_rate': trial.suggest_loguniform('learning_rate', min(tuning_config["learning_rate"]), max(tuning_config["learning_rate"])),
        'weight_decay': trial.suggest_loguniform('weight_decay', min(tuning_config["weight_decay"]), max(tuning_config["weight_decay"]))
    }
    return train_model(cfg, hyperparams, trial)

def fine_tune():
    logger.info("🔍 Starting Optuna tuning...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=load_tuning_config()["n_trials"])
    logger.info(f"🏆 Best trial: {study.best_trial.params} with val_loss={study.best_trial.value:.4f}")

#================= Inference ================ #
def load_inference_config(path='dl/config/inference_config.json'):
    with open(path, 'r') as f:
        return json.load(f)

def run_inference():
    logger.info("🧠 Running inference...")

    cfg = load_config()
    infer_cfg = load_inference_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg)
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(device)
    model.eval()

    data = read_csv_data(infer_cfg["data_path"], "Inference Data", is_target=False)
    dataset = WearableDataset(data, targets=None)
    loader = DataLoader(dataset, batch_size=infer_cfg.get("batch_size", 64), shuffle=False)

    predictions = []

    with torch.no_grad():
        for original_input, inputs, _ in tqdm(loader, desc="Inferring", colour="blue"):
            inputs = inputs.to(device)
            outputs = model((original_input.to(device), inputs))
            predictions.extend(outputs.cpu().numpy().flatten())

    # Save predictions to file
    output_path = infer_cfg.get("output_path", "inference_results.csv")
    df = pd.DataFrame({'Prediction': predictions})
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Inference complete. Results saved to: {output_path}")
    mlflow.log_artifact(output_path)

    # Optional: return for downstream use
    return predictions


# ============ Testing ============ #
def test_model(config, hyperparams):
    logger.info("🧪 Running final test on saved model...")
    test_data = read_csv_data('dataset/test_data.csv', "Test Data")
    test_targets = read_csv_data('dataset/test_target.csv', "Test Target", is_target=True)
    test_dataset = WearableDataset(test_data, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config)
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(device)
    model.eval()

    criterion = get_loss_function(config)
    test_loss, preds, labels = 0.0, [], []

    with torch.no_grad():
        for original_input, inputs, targets in tqdm(test_loader, desc="Testing", colour="cyan"):
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            outputs = model((original_input.to(device), inputs))
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            preds.extend(outputs.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    preds, labels = np.array(preds), np.array(labels)
    mae, mse = np.mean(np.abs(preds - labels)), np.mean((preds - labels)**2)
    logger.info(f"Test Loss: {test_loss/len(test_loader.dataset):.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}")
    mlflow.log_metrics({"test_loss": test_loss, "test_mae": mae, "test_mse": mse})

# ============ CLI ============ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'fine_tune', 'test', 'inference'], default='fine_tune')
    args = parser.parse_args()
    cfg = load_config()
    default_hyper = {
        "learning_rate": cfg.get("learning_rate", 1e-6),
        "batch_size": cfg.get("batch_size", 64),
        "weight_decay": cfg.get("weight_decay", 1e-5)
    }

    if args.mode == 'fine_tune':
        fine_tune()
    elif args.mode == 'test':
        test_model(cfg, default_hyper)
    elif args.mode == 'inference':
        run_inference()
    else:
        train_model(cfg, default_hyper)
