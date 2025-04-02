import pandas as pd
import numpy as np
import os
import logging
from colorlog import ColoredFormatter
from tqdm import tqdm

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
    }
)
handler.setFormatter(formatter)
logger = logging.getLogger('RespRateProcessor')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

BASE_DIR = "/media/data/Workspace/Respiratory-Rate-PPG-ECG/dataset/Resp/"
DEFAULT_HEADER = "Time,RelativeTime,SystemLocalTime,Unknown,NOM_PULS_OXIM_SAT_O2,NOM_PLETH_PULS_RATE,NOM_PULS_OXIM_PERF_REL,NOM_ECG_CARD_BEAT_RATE,NOM_RESP_RATE,NOM_ECG_AMPL_ST_I,NOM_ECG_AMPL_ST_II,NOM_ECG_AMPL_ST_III,NOM_ECG_AMPL_ST_AVR,NOM_ECG_AMPL_ST_AVL,NOM_ECG_AMPL_ST_AVF,NOM_ECG_AMPL_ST_V,NOM_ECG_AMPL_ST_MCL,NOM_ECG_V_P_C_CNT,NOM_PRESS_BLD_NONINV_SYS,NOM_PRESS_BLD_NONINV_DIA,NOM_PRESS_BLD_NONINV_MEAN"
DEFAULT_VALUES = "23-09-2023 10:05:24.480,556384768,23-09-2023 10:04:34.674,-,100,122,1.5,124,27,0.2,2.6,2.2,-1.5,-0.9,2.4,1.2,1.8,0,-,-,-" 

def get_monitor_files(file_path):
    monitor_files = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.csv') and 'monitorphillips-mpdataexport' in file.lower() and 'processed' not in file.lower():
                monitor_files.append(os.path.join(root, file))
    return monitor_files

def validate_milestones(values, ecg_idx=7, resp_idx=8):
    try:
        ecg_val = float(values[ecg_idx])
        resp_val = float(values[resp_idx])
        if 40 <= ecg_val <= 250 and 5 <= resp_val <= 50:
            return True
    except:
        pass
    return False

def adjust_row(values, previous_values, target_length, ecg_idx=7, resp_idx=8):
    shifts = len(values) - target_length
    
    fixed_cols = values[:3]
    adjustable_cols = values[3:]
    
    prev_ecg = float(previous_values[ecg_idx])
    prev_resp = float(previous_values[resp_idx])

    def is_valid(current_values):
        try:
            ecg_val = float(current_values[ecg_idx])
            resp_val = float(current_values[resp_idx])
            return (40 <= ecg_val <= 250 and 5 <= resp_val <= 50 and
                    abs(ecg_val - prev_ecg) <= 80 and
                    abs(resp_val - prev_resp) <= 15)
        except:
            return False

    # Case 1: No shifting needed
    if shifts == 0 and is_valid(values):
        return values

    # Case 2: Too many columns, shift left carefully removing "-" columns only
    if shifts > 0:
        for shift in range(shifts + 1):
            temp_cols = adjustable_cols.copy()
            removed = 0
            idx = 0
            while removed < shift and idx < len(temp_cols):
                if temp_cols[idx] == '-':
                    temp_cols.pop(idx)
                    removed += 1
                else:
                    idx += 1

            if len(temp_cols) == target_length - 3:
                test_values = fixed_cols + temp_cols
                if is_valid(test_values):
                    return test_values

    # Case 3: Too few columns, consider padding left or right
    elif shifts < 0:
        pad_len = abs(shifts)

        # Padding on right (end)
        padded_right = adjustable_cols + ['-'] * pad_len
        test_values_right = fixed_cols + padded_right
        if is_valid(test_values_right):
            return test_values_right

        # Padding on left (start from 4th col)
        padded_left = ['-'] * pad_len + adjustable_cols
        test_values_left = fixed_cols + padded_left[:target_length-3]
        if is_valid(test_values_left):
            return test_values_left

    # No valid adjustment found
    return None

def process_monitor_data(file_path):
    try:
        monitor_files = get_monitor_files(file_path)
        if not monitor_files:
            logger.warning(f"No monitor files found in {file_path}")
            return

        default_values_list = DEFAULT_VALUES.split(',')

        # Numeric difference calculation for first-row shifting
        def numeric_diff(row_a, row_b):
            diff = 0
            for a, b in zip(row_a[3:], row_b[3:]):
                try:
                    diff += abs(float(a) - float(b))
                except:
                    continue
            return diff

        for monitor_file in tqdm(monitor_files, desc="Processing monitor files"):
            logger.info(f"Processing {monitor_file}")

            with open(monitor_file, 'r', encoding='utf-8') as f:
                content = f.readlines()

            if content and content[0].strip() != DEFAULT_HEADER:
                logger.info(f"Replacing header in {monitor_file}")
                content[0] = DEFAULT_HEADER + '\n'

            processed_rows = []
            previous_row = None
            holding_row = None  # Temporarily hold the first invalid-length-matching row

            for row in tqdm(content[1:], desc=f"Rows in {os.path.basename(monitor_file)}", leave=False):
                row = row.strip()
                if not row:
                    continue

                values = row.split(',')

                # First row special handling
                if previous_row is None:
                    if len(values) == len(default_values_list):
                        if validate_milestones(values):
                            previous_row = values
                            processed_rows.append(','.join(values))
                        else:
                            logger.warning("First row milestones invalid but length matches; holding this row temporarily.")
                            holding_row = values  # Hold the row temporarily
                    else:
                        # Handle mismatch in length by shifting
                        min_diff = float('inf')
                        best_values = None
                        shifts = len(values) - len(default_values_list)
                        adjustable_cols = values[3:]

                        # Shifting left or right logic
                        if shifts > 0:
                            for shift in range(shifts + 1):
                                test_cols = adjustable_cols[shift:shift + len(default_values_list) - 3]
                                test_values = values[:3] + test_cols
                                diff = numeric_diff(test_values, default_values_list)
                                if diff < min_diff:
                                    min_diff = diff
                                    best_values = test_values
                        elif shifts < 0:
                            pad = ['-'] * abs(shifts)
                            # Pad left
                            test_values_left = values[:3] + pad + adjustable_cols
                            test_values_left = test_values_left[:len(default_values_list)]
                            diff_left = numeric_diff(test_values_left, default_values_list)
                            if diff_left < min_diff:
                                min_diff = diff_left
                                best_values = test_values_left
                            # Pad right
                            test_values_right = values[:3] + adjustable_cols + pad
                            test_values_right = test_values_right[:len(default_values_list)]
                            diff_right = numeric_diff(test_values_right, default_values_list)
                            if diff_right < min_diff:
                                min_diff = diff_right
                                best_values = test_values_right

                        if best_values and validate_milestones(best_values):
                            previous_row = best_values
                            processed_rows.append(','.join(best_values))
                            logger.info("First row adjusted successfully using minimal difference shifting.")
                        else:
                            logger.warning("First row anomaly unresolved; using default values as fallback.")
                            previous_row = default_values_list
                            processed_rows.append(','.join(previous_row))
                else:
                    # Validate subsequent rows against previous row
                    if len(values) != len(default_values_list) or not validate_milestones(values):
                        adjusted_values = adjust_row(values, previous_row, len(default_values_list))
                        if adjusted_values:
                            values = adjusted_values
                        else:
                            logger.debug("Row anomaly unresolved, reverting to previous valid row.")
                            values = previous_row
                    previous_row = values
                    processed_rows.append(','.join(values))

                    # If we have a valid row now and a holding_row exists, process the holding_row
                    if holding_row:
                        logger.info("Appending previously held row now that we have a valid baseline.")
                        processed_rows.insert(0, ','.join(holding_row))
                        holding_row = None  # Reset holding_row after appending

            # Write processed data to file
            output_file = monitor_file.replace('.csv', '_processed.csv')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content[0])
                f.write('\n'.join(processed_rows))

            logger.info(f"Processed file saved as {output_file}")

    except Exception as e:
        logger.error(f"Error processing monitor data: {str(e)}")
        raise


if __name__ == "__main__":
    process_monitor_data(BASE_DIR)
