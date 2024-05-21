import plotly.graph_objects as go
import numpy as np
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pywt
import pandas as pd

def bandpass_filter(signal, fs, lowcut=0.1, highcut=0.5, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def wavelet_denoise(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, mode='periodization')
    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = coeffs[:]
    denoised_coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in denoised_coeffs[1:]]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet, mode='periodization')
    return denoised_signal

def preprocess_signal(signal, fs):
    filtered_signal = bandpass_filter(signal, fs)
    denoised_signal = wavelet_denoise(filtered_signal)
    return denoised_signal

def remove_heartbeat_trend(signal, fs):
    # Apply a low-pass filter to remove high-frequency components (heartbeat)
    b, a = butter(3, 0.5 / (0.5 * fs), btype='low')
    respiratory_signal = filtfilt(b, a, signal)
    return respiratory_signal

def detect_respiratory_peaks(signal, fs):
    # Remove heartbeat trend to isolate respiratory signal
    respiratory_signal = remove_heartbeat_trend(signal, fs)
    
    # Find peaks in the respiratory signal
    distance = int(1.5 * fs)  # Assuming a minimum respiratory rate of 40 breaths per minute
    peaks, _ = find_peaks(respiratory_signal, distance=distance, 
                          height=np.mean(respiratory_signal) - np.std(respiratory_signal))
    # peaks, _ = find_peaks(respiratory_signal, distance=distance, height=np.mean(respiratory_signal) + 0.5 * np.std(respiratory_signal))
    
    return peaks


def calculate_respiratory_rate(peaks, fs, duration):
    if len(peaks) < 2:
        return 0  # Not enough peaks to calculate a respiratory rate
    # Calculate the respiratory rate in breaths per minute
    rr_bpm = (len(peaks) - 1) * 60 / duration
    return rr_bpm



def get_rr(signal, fs, preprocess=True):
    if preprocess:
        signal = preprocess_signal(signal, fs)
    
    # Detect peaks corresponding to the respiratory cycles
    respiratory_peaks = detect_respiratory_peaks(signal, fs)
    
    # Calculate the duration of the signal in seconds
    duration = len(signal) / fs
    
    # Calculate the respiratory rate from the detected peaks
    rr_bpm = calculate_respiratory_rate(respiratory_peaks, fs, duration)
    
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

