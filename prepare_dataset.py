import numpy as np
import logging
from colorlog import ColoredFormatter

# Setup colored logging
handler = logging.StreamHandler()
formatter = ColoredFormatter(
    "%(log_color)s[%(levelname)s]%(reset)s %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    })
handler.setFormatter(formatter)
logger = logging.getLogger('DataVerificationSplit')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Paths
DATA_PATH = "/media/data/Workspace/Respiratory-Rate-PPG-ECG/dataset/Resp/Final_Compiled_Data/final_train_data.csv"
TARGET_PATH = "/media/data/Workspace/Respiratory-Rate-PPG-ECG/dataset/Resp/Final_Compiled_Data/final_train_target.csv"

# Output paths
TRAIN_DATA_PATH = "train_data_verified.csv"
TRAIN_TARGET_PATH = "train_target_verified.csv"
TEST_DATA_PATH = "test_data_verified.csv"
TEST_TARGET_PATH = "test_target_verified.csv"

# Load data
def load_data(data_path, target_path):
    data = np.loadtxt(data_path, delimiter=",")
    target = np.loadtxt(target_path, delimiter=",")

    if target.ndim == 1:
        target = target.reshape(-1, 1)

    logger.info(f"Loaded data shape: {data.shape}")
    logger.info(f"Loaded target shape: {target.shape}")

    return data, target

# Verification step
def verify_data(data, target):
    logger.info("Starting verification process...")

    # Step 1: Remove rows with target > 60
    valid_indices = np.where(target.flatten() <= 60)[0]
    removed_high_rate = len(target) - len(valid_indices)
    data, target = data[valid_indices], target[valid_indices]

    logger.info(f"Removed {removed_high_rate} rows with breathing rate > 60.")

    # Step 2: Remove duplicate rows in data
    data_unique, unique_indices = np.unique(data, axis=0, return_index=True)
    duplicate_count = len(data) - len(data_unique)

    target_unique = target[unique_indices]

    logger.info(f"Removed {duplicate_count} duplicate rows from data.")
    logger.info(f"Verified data shape: {data_unique.shape}")
    logger.info(f"Verified target shape: {target_unique.shape}")

    return data_unique, target_unique

# Shuffle and split data
def shuffle_and_split(data, target, train_ratio=0.8):
    logger.info("Shuffling and splitting data...")

    # Shuffle
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data, target = data[indices], target[indices]

    # Split
    split_idx = int(len(data) * train_ratio)
    train_data, test_data = data[:split_idx], data[split_idx:]
    train_target, test_target = target[:split_idx], target[split_idx:]

    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Testing data shape: {test_data.shape}")

    return train_data, train_target, test_data, test_target

# Save verified data
def save_data(data, target, data_path, target_path):
    np.savetxt(data_path, data, delimiter=",", fmt='%.5f')
    np.savetxt(target_path, target, delimiter=",", fmt='%d')

    logger.info(f"Saved data to '{data_path}'")
    logger.info(f"Saved target to '{target_path}'")

# Main workflow
def main():
    data, target = load_data(DATA_PATH, TARGET_PATH)

    data_verified, target_verified = verify_data(data, target)

    train_data, train_target, test_data, test_target = shuffle_and_split(
        data_verified, target_verified, train_ratio=0.8
    )

    # Save final datasets
    save_data(train_data, train_target, TRAIN_DATA_PATH, TRAIN_TARGET_PATH)
    save_data(test_data, test_target, TEST_DATA_PATH, TEST_TARGET_PATH)

if __name__ == "__main__":
    main()
