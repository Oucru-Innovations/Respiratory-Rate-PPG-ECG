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

BASE_DIR = "/media/data/Workspace/Respiratory-Rate-PPG-ECG/dataset/Resp"
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

def compile_patient_data(patient_folder, output_folder):
    final_df = pd.DataFrame()
    patient_id = os.path.basename(patient_folder)

    date_folders = [
        date_folder for date_folder in glob.glob(os.path.join(patient_folder, "*"))
        if os.path.isdir(os.path.join(date_folder, "PPG")) and os.path.isdir(os.path.join(date_folder, "Monitor"))
    ]

    for date_folder in tqdm(date_folders, desc=f"Processing dates for {patient_id}", leave=False):
        ppg_files = glob.glob(os.path.join(date_folder, "PPG", "SmartCareCsv_*.csv"))
        monitor_files = glob.glob(os.path.join(date_folder, "Monitor", "*MonitorPhillips-MPDataExport*processed*.csv"))

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
        os.makedirs(output_folder, exist_ok=True)
        final_output = os.path.join(output_folder, f"{patient_id}_mapped.csv")
        final_df.to_csv(final_output, index=False)
        logger.info(f"Patient data compilation complete. Saved to '{final_output}' ({len(final_df)} rows).")

        # Create training data for the patient
        pleth_output_path = os.path.join(output_folder, "train_data.csv")
        target_output_path = os.path.join(output_folder, "train_target.csv")
        create_training_data(final_output, pleth_output_path, target_output_path)
    else:
        logger.error(f"No data compiled for patient {patient_id}. Check input files.")


def compile_all_patients(base_dir, final_output_folder):
    patient_folders = glob.glob(os.path.join(base_dir, "*"))

    # Compile data patient-by-patient
    for patient_folder in tqdm(patient_folders, desc="Compiling patient data"):
        patient_output_folder = os.path.join(patient_folder, "compiled_data")
        compile_patient_data(patient_folder, patient_output_folder)

    # After individual processing, compile all patient data
    all_train_data = []
    all_train_target = []

    # Gather all patient-level compiled data
    for patient_folder in tqdm(patient_folders, desc="Aggregating all patients"):
        compiled_data_folder = os.path.join(patient_folder, "compiled_data")
        train_data_path = os.path.join(compiled_data_folder, "train_data.csv")
        train_target_path = os.path.join(compiled_data_folder, "train_target.csv")

        if os.path.exists(train_data_path) and os.path.exists(train_target_path):
            patient_train_data = np.loadtxt(train_data_path, delimiter=",")
            patient_train_target = np.loadtxt(train_target_path, delimiter=",")
            
            # Ensure 2D shape for targets
            if patient_train_target.ndim == 1:
                patient_train_target = patient_train_target.reshape(-1, 1)

            all_train_data.append(patient_train_data)
            all_train_target.append(patient_train_target)
        else:
            logger.warning(f"Missing training data for patient folder: {patient_folder}")

    # Combine all data if not empty
    if all_train_data and all_train_target:
        combined_train_data = np.vstack(all_train_data)
        combined_train_target = np.vstack(all_train_target)

        # Save combined data
        os.makedirs(final_output_folder, exist_ok=True)
        combined_data_path = os.path.join(final_output_folder, "final_train_data.csv")
        combined_target_path = os.path.join(final_output_folder, "final_train_target.csv")

        np.savetxt(combined_data_path, combined_train_data, delimiter=",", fmt='%.5f')
        np.savetxt(combined_target_path, combined_train_target, delimiter=",", fmt='%d')

        logger.info(f"Final training data compiled successfully.")
        logger.info(f"Combined train data shape: {combined_train_data.shape}, saved to '{combined_data_path}'")
        logger.info(f"Combined train target shape: {combined_train_target.shape}, saved to '{combined_target_path}'")
    else:
        logger.error("No training data available for compilation.")

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
    final_output_folder = os.path.join(BASE_DIR, "Final_Compiled_Data")
    logger.info("Starting patient-level data compilation process...")
    compile_all_patients(BASE_DIR, final_output_folder)


if __name__ == "__main__":
    main()
