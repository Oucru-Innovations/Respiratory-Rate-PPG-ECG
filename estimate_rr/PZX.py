import numpy as np
import pandas as pd
from scipy import signal
import sys
sys.path.append(".")
from preprocess.preprocess import preprocess_signal

def calculate_respiratory_rate_from_zero_crossings(zero_crossings, fs, duration):
    if len(zero_crossings) < 2:
        return 0  # Not enough zero-crossings to calculate a respiratory rate
    # Calculate the respiratory rate in breaths per minute
    rr_bpm = (len(zero_crossings) - 1) * 60 / duration
    return rr_bpm

def detect_zero_crossings(sig):
    zero_crossings = np.where(np.diff(np.sign(sig)))[0]
    return zero_crossings


def get_rr(sig, fs, signal_type='ECG',preprocess=True):
    if preprocess:
        sig = preprocess_signal(sig, fs, signal_type)
    
    # Detect zero-crossings corresponding to the respiratory cycles
    zero_crossings = detect_zero_crossings(sig)
    
    # Calculate the duration of the signal in seconds
    duration = len(sig) / fs
    
    # Calculate the respiratory rate from the detected zero-crossings
    rr_bpm = calculate_respiratory_rate_from_zero_crossings(zero_crossings, fs, duration)
    
    return rr_bpm


def get_sign(extrema_indices, trough_indices, peak_indices):
    if extrema_indices in trough_indices:
        return -1
    return 1

# if __name__ == "__main__":

#     calculated_fs = 256
#     ecg_data_path = "dataset/public_ecg_data.csv"
#     ecg_target_path = "dataset/public_ecg_target.csv"
    
#     # calculated_fs = 100
#     # ecg_data_path = "dataset/public_ppg_data.csv"
#     # ecg_target_path = "dataset/public_ppg_target.csv"
#     # Load the ECG data and target files again, ensuring correct parsing
#     ecg_data = pd.read_csv(ecg_data_path, header=None)
#     ecg_target = pd.read_csv(ecg_target_path, header=None)

#     # Display the shape and first few rows of the data and target files
#     ecg_data_shape = ecg_data.shape
#     ecg_target_shape = ecg_target.shape

#     ecg_data_head = ecg_data.head()
#     ecg_target_head = ecg_target.head()
    
#     target_rr = ecg_target.values.flatten()

#     # Apply the estimate_rr_peaks function to each segment
#     estimated_rr = []
#     for index, row in ecg_data.iterrows():
#         segment = row.values
#         rr = get_rr(segment, calculated_fs)
#         estimated_rr.append(rr)

#     # Filter out None values from estimated_rr
#     valid_estimates = [(est, tgt) for est, tgt in zip(estimated_rr, target_rr) if est is not None]
#     estimated_rr_valid, target_rr_valid = zip(*valid_estimates)

#     # Convert to numpy arrays for easier comparison
#     estimated_rr_valid = np.array(estimated_rr_valid)
#     target_rr_valid = np.array(target_rr_valid)

#     # Calculate the Mean Absolute Error (MAE) as a simple metric of accuracy
#     mae = np.mean(np.abs(estimated_rr_valid - target_rr_valid))
#     print(mae)
#     print(np.round(estimated_rr_valid[:100]))
#     print( target_rr_valid[:100])  # Display MAE and first 10 estimated vs. target values for verification
