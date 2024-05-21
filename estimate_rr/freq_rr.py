from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import resample
import numpy as np

import numpy as np
from scipy.signal import butter, filtfilt
import pywt
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

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

def calculate_rr_from_ar_poles(sig, fs, order=8, method='max_power'):

    # Fit autoregressive model
    ar_model = AutoReg(sig, lags=order).fit()
    
    # Get the AR coefficients
    ar_params = np.r_[1, -ar_model.params]  # Include lag 0 coefficient
    
    # Find the poles of the AR model
    poles = np.roots(ar_params)
    
    # Calculate the angles of the poles
    angles = np.angle(poles)
    
    # Convert angles to normalized frequencies (cycles/sample)
    frequencies_norm = angles * fs / (2 * np.pi)
    
    # Convert normalized frequencies to frequencies in Hz
    frequencies_hz = frequencies_norm / (fs/2)
    
    # Convert frequencies from Hz to breaths per minute (BPM)
    frequencies_bpm = np.abs(frequencies_hz) * 60
    
    # Select the dominant frequency within the respiratory range (6 to 30 BPM)
    respiratory_frequencies = frequencies_bpm[(frequencies_bpm > 6) & (frequencies_bpm < 40)]
    if len(respiratory_frequencies) == 0:
        return 0
    
    if method == 'max':
        dominant_frequency = np.max(respiratory_frequencies)
    elif method == 'median':
        dominant_frequency = np.median(respiratory_frequencies)
    elif method == 'mean':
        dominant_frequency = np.mean(respiratory_frequencies)
    elif method == 'max_power':
        # Calculate power of each frequency
        powers = np.abs(np.abs(poles) ** 2)
        dominant_frequency = respiratory_frequencies[np.argmax(powers)]
    
    return dominant_frequency

def get_rr(signal, fs, preprocess=True, order=10):
    if preprocess:
        signal = preprocess_signal(signal, fs)
    
    rr_bpm = calculate_rr_from_ar_poles(signal, fs, order=order)
    
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