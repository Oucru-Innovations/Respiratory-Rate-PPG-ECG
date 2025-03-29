import json
import torch
from torch.utils.data import DataLoader
from model_pool import ViTRegressor
from wearable_data import WearableDataset
from model_utils import loss_uniform_spread_l2
import torch.optim as optim
import os, sys
import numpy as np
import mlflow
import argparse
import optuna
import logging
import colorlog
from tqdm import tqdm

# Setup colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s | %(levelname)s | %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

CONFIG_PATH = 'dl/config/config.json'
CONFIG_TUNING_PATH = 'dl/config/config_tuning.json'

def load_config():
    config_path=CONFIG_PATH
    with open(config_path, 'r') as f:
        return json.load(f)

def load_tuning_config():
    tuning_config_path=CONFIG_TUNING_PATH
    with open(tuning_config_path, 'r') as f:
        return json.load(f)

def read_csv_data(file_path, description, is_target=False):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{description} CSV file not found at {file_path}")
        data = np.loadtxt(file_path, delimiter=',')
        if is_target:
            data = data.reshape(-1, 1)  # explicitly reshape targets
    except Exception as e:
        print(f"Error reading {description} CSV: {e}")
        sys.exit(1)
    return data


def train_model(config, hyperparams, trial=None):
    logger.info("Loading datasets...")
    train_data = read_csv_data('dataset/train_data.csv', "Training data")
    train_targets = read_csv_data('dataset/train_target.csv', "Training target", is_target=True)
    val_data = read_csv_data('dataset/val_data.csv', "Validation data")
    val_targets = read_csv_data('dataset/val_target.csv', "Validation target", is_target=True)

    train_dataset = WearableDataset(train_data, train_targets)
    val_dataset = WearableDataset(val_data, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTRegressor(fs=config["fs_ecg"] if config["signal_type"]=="ecg" else config["fs_ppg"])
    model.to(device)

    criterion = loss_uniform_spread_l2()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])

    mlflow.start_run()
    mlflow.log_params({**config, **hyperparams})

    for epoch in tqdm(range(1, config["epochs"] + 1), desc='Epochs', colour='green'):
        model.train()
        train_loss = 0.0
        for original_input, inputs, targets in tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=False, colour='blue'):
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model((original_input.to(device), inputs))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        logger.info(f'Epoch [{epoch}/{config["epochs"]}] - Train Loss: {train_loss:.4f}')
        mlflow.log_metric('train_loss', train_loss, step=epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for original_input, inputs, targets in tqdm(val_loader, desc=f'Val Epoch {epoch}', leave=False, colour='magenta'):
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                outputs = model((original_input.to(device), inputs))
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        logger.info(f'Epoch [{epoch}/{config["epochs"]}] - Val Loss: {val_loss:.4f}')
        mlflow.log_metric('val_loss', val_loss, step=epoch)

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                logger.warning("Trial was pruned by Optuna.")
                mlflow.end_run()
                raise optuna.exceptions.TrialPruned()

    torch.save(model.state_dict(), 'best_model.pth')
    mlflow.log_artifact('best_model.pth')
    mlflow.end_run()

    logger.info("Training complete. Best model saved.")

    return val_loss


def objective(trial):
    tuning_config = load_tuning_config()

    hyperparams = {
        'batch_size': trial.suggest_categorical('batch_size', tuning_config["batch_size"]),
        'learning_rate': trial.suggest_loguniform('learning_rate', min(tuning_config["learning_rate"]), max(tuning_config["learning_rate"])),
        'weight_decay': trial.suggest_loguniform('weight_decay', min(tuning_config["weight_decay"]), max(tuning_config["weight_decay"]))
    }

    base_config = {
        "epochs": tuning_config["epochs"],
        "signal_type": tuning_config["signal_type"],
        "fs_ecg": tuning_config["fs_ecg"],
        "fs_ppg": tuning_config["fs_ppg"]
    }

    val_loss = train_model(base_config, hyperparams, trial)
    return val_loss

def fine_tune():
    tuning_config = load_tuning_config()
    logger.info("Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction='minimize', 
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=tuning_config["n_trials"])

    logger.info(f"Best trial parameters: {study.best_trial.params}")
    logger.info(f"Best trial validation loss: {study.best_trial.value:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pipeline with optional fine-tuning (Optuna).')
    parser.add_argument('--mode', choices=['train', 'eval', 'fine_tune'], default='train')
    args = parser.parse_args()

    config = load_config()

    if args.mode == 'fine_tune':
        fine_tune()
    else:
        default_hyperparams = {
            'learning_rate': config.get('learning_rate', 1e-4),
            'batch_size': config.get('batch_size', 64),
            'weight_decay': config.get('weight_decay', 1e-5)
        }
        train_model(config, default_hyperparams)
