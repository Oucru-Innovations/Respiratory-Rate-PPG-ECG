import numpy as np
from scipy import signal
import os
from scipy.signal import butter, filtfilt, argrelmax, argrelmin, find_peaks
# from preprocess.preprocess_signal import preprocess_signal
import pandas as pd
# from preprocess.preprocess import preprocess_signal
import pywt

def bandpass_filter(signal, fs, lowcut=0.1, highcut=0.5, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def wavelet_denoise(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, mode='periodization')
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = coeffs[:]
    denoised_coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in denoised_coeffs[1:]]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet, mode='periodization')
    return denoised_signal

def preprocess_signal(signal, fs):
    filtered_signal = bandpass_filter(signal, fs)
    denoised_signal = wavelet_denoise(filtered_signal)
    return denoised_signal

def get_valid_rr(sig, local_min, local_max, thres):
    resp_markers = []
    rel_peaks = local_max[sig[local_max] > thres]
    rel_troughs = local_min[sig[local_min] < 0]
    for i in range(len(rel_peaks) - 1):
        cyc_rel_troughs = (np.where((rel_troughs > rel_peaks[i]) &
                                    (rel_troughs < rel_peaks[i + 1])))[0]
        if len(cyc_rel_troughs) == 1:
            resp_markers.append((rel_peaks[i], rel_peaks[i + 1]))
    return resp_markers

def get_rr(sig, fs, preprocess=True):
    if preprocess:
        sig = preprocess_signal(sig, fs)
    
    local_max = argrelmax(sig, order=4)[0]
    local_min = argrelmin(sig, order=4)[0]

    # local_max = find_peaks(sig, distance=fs/6, height=np.mean(sig)*1. + np.std(sig))[0]
    # local_min = find_peaks(-sig, distance=fs/6, height=np.mean(sig)*0.15 + np.std(sig))[0]
    
    # max_threshold = np.quantile(sig[local_max], 0.5) * 0.5
    max_threshold = 0

    resp_markers = get_valid_rr(sig, local_min, local_max, max_threshold)

    last_n_peaks = 3
    if len(resp_markers) < last_n_peaks:
        return 0  # Not enough cycles to calculate RR using the last 5 cycles
    
    # Extract the times of the last n peaks
    last_peaks = np.array([marker[1] for marker in resp_markers[-last_n_peaks-1:]])
    
    # Calculate the time differences between the last 5 adjacent peaks
    peak_intervals = np.diff(last_peaks) / fs  # Convert samples to seconds
    
    # Compute the mean interval
    mean_interval = np.mean(peak_intervals)
    
    # Convert the mean interval to respiratory rate in BPM
    rr_bpm = 60 / mean_interval  # Calculate BPM
    
    return rr_bpm

if __name__ == "__main__":

    calculated_fs = 256
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

