import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pywt
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d
import sys
sys.path.append('.')
from preprocess.preprocess import preprocess_signal

def detect_peaks(signal, fs, min_rr=6, max_rr=30):
    # Apply preprocessing to the signal
    preprocessed_signal = preprocess_signal(signal, fs, signal_type='PPG')
    
    # Define the distance based on expected respiratory rate range
    min_distance = int(fs * 60 / max_rr)
    max_distance = int(fs * 60 / min_rr)
    
    # Dynamic peak detection with adaptive thresholding
    peaks, _ = find_peaks(preprocessed_signal, distance=min_distance, height=None)
    
    if len(peaks) < 2:
        height_threshold = np.mean(preprocessed_signal) + 0.2 * np.std(preprocessed_signal)
        peaks, _ = find_peaks(preprocessed_signal, distance=min_distance, height=height_threshold)
    
    if len(peaks) < 2:
        height_threshold = np.mean(preprocessed_signal) + 0.1 * np.std(preprocessed_signal)
        peaks, _ = find_peaks(preprocessed_signal, distance=min_distance, height=height_threshold)
    
    # Filter out peaks that are too close to each other
    filtered_peaks = []
    for i in range(len(peaks) - 1):
        if min_distance <= (peaks[i+1] - peaks[i]) <= max_distance:
            filtered_peaks.append(peaks[i])
    if len(peaks) > 0:
        filtered_peaks.append(peaks[-1])  # Include the last peak if valid
    
    return np.array(filtered_peaks)

def calculate_respiratory_rate_from_cta(peaks, fs):
    if len(peaks) < 2:
        return 0  # Not enough peaks to calculate a respiratory rate

    # Calculate the intervals between peaks
    intervals = np.diff(peaks) / fs  # Convert from samples to seconds
    
    # Filter out unrealistic intervals
    min_interval = 60 / 30  # Maximum respiratory rate: 30 breaths per minute
    max_interval = 60 / 6   # Minimum respiratory rate: 6 breaths per minute
    valid_intervals = intervals[(intervals >= min_interval) & (intervals <= max_interval)]
    
    if len(valid_intervals) == 0:
        return 0
    
    # Calculate the mean interval
    mean_interval = np.mean(valid_intervals)
    
    # Convert the mean interval to respiratory rate in breaths per minute
    rr_bpm = 60 / mean_interval
    
    return rr_bpm

def get_rr(signal, fs, signal_type='ECG', preprocess=True):
    if preprocess:
        signal = preprocess_signal(signal, fs, signal_type)
    
    # Detect peaks in the signal
    peaks = detect_peaks(signal, fs)
    
    # Calculate the respiratory rate from the detected peaks
    rr_bpm = calculate_respiratory_rate_from_cta(peaks, fs)
    
    return rr_bpm


if __name__ == "__main__":
    calculated_fs = 256
    signal_type='ECG'
    ecg_data_path = "dataset/public_ecg_data.csv"
    ecg_target_path = "dataset/public_ecg_target.csv"
    # Load the ECG data and target files again, ensuring correct parsing
    ecg_data = pd.read_csv(ecg_data_path, header=None)
    ecg_target = pd.read_csv(ecg_target_path, header=None)

    # Display the shape and first few rows of the data and target files
    ecg_data_shape = ecg_data.shape
    ecg_target_shape = ecg_target.shape

    ecg_data_head = ecg_data.head()
    ecg_target_head = ecg_target.head()
    
    target_rr = ecg_target.values.flatten()

    # Apply the estimate_rr_peaks function to each segment
    estimated_rr = []
    for index, row in ecg_data.iterrows():
        segment = row.values
        rr = get_rr(segment, calculated_fs)
        estimated_rr.append(rr)

    # Filter out None values from estimated_rr
    valid_estimates = [(est, tgt) for est, tgt in zip(estimated_rr, target_rr) if est is not None]
    estimated_rr_valid, target_rr_valid = zip(*valid_estimates)

    # Convert to numpy arrays for easier comparison
    estimated_rr_valid = np.array(estimated_rr_valid)
    target_rr_valid = np.array(target_rr_valid)

    # Calculate the Mean Absolute Error (MAE) as a simple metric of accuracy
    mae = np.mean(np.abs(estimated_rr_valid - target_rr_valid))
    print(mae)
    print(np.round(estimated_rr_valid[:100]))
    print( target_rr_valid[:100])  # Display MAE and first 10 estimated vs. target values for verification
