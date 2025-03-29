import os
import pandas as pd
import ast
import numpy as np
import glob
from tqdm import tqdm
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
logger = logging.getLogger('RespRateProcessor')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

BASE_DIR = "/media/data/Workspace/Respiratory-Rate-PPG-ECG/dataset/Resp/24EIb-003-029/"
FINAL_OUTPUT = "final_mapped_resp_rate_full.csv"

def process_and_combine_pleth(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='mixed').dt.tz_convert('Asia/Ho_Chi_Minh')

        def safe_eval(x):
            try:
                return ast.literal_eval(x) if isinstance(x, str) else x
            except:
                return None

        df['pleth'] = df['pleth'].apply(safe_eval)
        df.dropna(subset=['pleth'], inplace=True)

        segments = []
        for i in range(0, len(df), 60):
            chunk = df.iloc[i:i+60]
            if len(chunk) == 60:
                segments.append({
                    'start_timestamp': chunk['timestamp'].iloc[0],
                    'end_timestamp': chunk['timestamp'].iloc[-1],
                    'pleth': [val for sublist in chunk['pleth'] for val in sublist]
                })
        return pd.DataFrame(segments)

    except Exception as e:
        logger.error(f"Error processing pleth file '{file_path}': {e}")
        return pd.DataFrame()

def map_closest_resp_rate(processed_df, comparison_df, file_path):
    try:
        comparison_df['SystemLocalTime'] = pd.to_datetime(
            comparison_df['SystemLocalTime'], 
            format='mixed'
        ).dt.tz_localize('Asia/Ho_Chi_Minh', ambiguous='NaT', nonexistent='shift_forward')

        comparison_df.dropna(subset=['SystemLocalTime'], inplace=True)
        comparison_df.sort_values('SystemLocalTime', inplace=True)
        comparison_df.reset_index(drop=True, inplace=True)

        def find_closest_resp_rate(target_time):
            idx = comparison_df['SystemLocalTime'].searchsorted(target_time)
            if idx == len(comparison_df):
                idx -= 1
            elif idx > 0:
                prev_idx, next_idx = idx - 1, idx
                if abs(comparison_df.loc[prev_idx, 'SystemLocalTime'] - target_time) < abs(comparison_df.loc[next_idx, 'SystemLocalTime'] - target_time):
                    idx = prev_idx
            return comparison_df.loc[idx, 'NOM_RESP_RATE']

        processed_df['NOM_RESP_RATE'] = processed_df['end_timestamp'].apply(find_closest_resp_rate)
        return processed_df

    except Exception as e:
        logger.error(f"Error mapping resp rate for file '{file_path}': {e}")
        return pd.DataFrame()

def compile_all_data(base_dir, final_output):
    final_df = pd.DataFrame()
    date_folders = glob.glob(os.path.join(base_dir, "*"))

    for date_folder in tqdm(date_folders, desc="Processing folders"):
        ppg_files = glob.glob(os.path.join(date_folder, "PPG", "SmartCareCsv_*.csv"))
        monitor_files = glob.glob(os.path.join(date_folder, "Monitor", "MonitorPhillips-MPDataExport*.csv"))

        if not ppg_files or not monitor_files:
            logger.warning(f"Missing PPG or Monitor files in folder: {date_folder}")
            continue

        for ppg_file in ppg_files:
            logger.info(f"Processing PPG file: {ppg_file}")
            processed_df = process_and_combine_pleth(ppg_file)
            if processed_df.empty:
                logger.warning(f"Empty processed dataframe for file: {ppg_file}")
                continue

            monitor_file = monitor_files[0]
            logger.info(f"Loading Monitor file: {monitor_file}")

            try:
                comparison_df = pd.read_csv(monitor_file, low_memory=False)
            except Exception as e:
                logger.error(f"Failed to read monitor file '{monitor_file}': {e}")
                continue

            mapped_df = map_closest_resp_rate(processed_df, comparison_df, ppg_file)

            if mapped_df.empty:
                logger.warning(f"Empty mapped dataframe for file: {ppg_file}")
                continue

            final_df = pd.concat([final_df, mapped_df], ignore_index=True)

    if not final_df.empty:
        final_df.to_csv(final_output, index=False)
        logger.info(f"Data compilation complete. Final data saved to '{final_output}' ({len(final_df)} rows).")
    else:
        logger.error("No data compiled. Please check input directories and files.")


def create_training_data(mapped_df_path, pleth_output_path, target_output_path):
    try:
        mapped_df = pd.read_csv(mapped_df_path)

        pleth_data = mapped_df['pleth'].apply(lambda x: np.array(ast.literal_eval(x)))
        train_data = np.vstack(pleth_data.values)
        train_target = mapped_df['NOM_RESP_RATE'].values.reshape(-1, 1)

        np.savetxt(pleth_output_path, train_data, delimiter=",", fmt='%.5f')
        np.savetxt(target_output_path, train_target, delimiter=",", fmt='%d')

        logger.info(f"Training data created successfully.")
        logger.info(f"Train data shape: {train_data.shape}, saved to '{pleth_output_path}'")
        logger.info(f"Train target shape: {train_target.shape}, saved to '{target_output_path}'")

    except Exception as e:
        logger.error(f"Error creating training data: {e}")

def main():
    logger.info("Starting data compilation process...")
    compile_all_data(BASE_DIR, FINAL_OUTPUT)

    pleth_output_path = "train_data.csv"
    target_output_path = "train_target.csv"
    create_training_data(FINAL_OUTPUT, pleth_output_path, target_output_path)

if __name__ == "__main__":
    main()
