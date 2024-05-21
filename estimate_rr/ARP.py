import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, argrelmax, find_peaks, welch
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal.windows import get_window
import pywt
import scipy
from scipy.signal import stft

def preprocess_signal(sig, fs, highpass=0.1, lowpass=0.9, order=3):
    nyquist = 0.5 * fs
    low = highpass / nyquist
    high = lowpass / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    filtered_sig = filtfilt(b, a, sig)
    return filtered_sig

def wavelet_denoise(sig, wavelet='db4', level=3):
    coeffs = pywt.wavedec(sig, wavelet, mode='periodization')
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
    denoised_coeffs = coeffs[:]
    denoised_coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in denoised_coeffs[1:]]
    denoised_sig = pywt.waverec(denoised_coeffs, wavelet, mode='periodization')
    return denoised_sig

def estimate_rr_arp(sig, fs,
                    window_type='hann', nperseg=1028, noverlap=128,
                    preprocess=True, use_wavelet=True,
                    lower_bound = 0.12,  # Corresponds to about 6 breaths per minute
                    upper_bound = 0.6  # Corresponds to about  breaths per minute
                    ):
    if preprocess:
        sig = preprocess_signal(sig, fs)
    if use_wavelet:
        sig = wavelet_denoise(sig)
    
    nperseg = fs*25
    noverlap = int(fs/4)
    window = get_window(window_type, nperseg)
    # Compute STFT
    f, t, Zxx = stft(sig, fs, window=window, nperseg=nperseg, noverlap=noverlap)
    
    resp_band = (f >= lower_bound) & (f <= upper_bound)
    
    # Sum the power in the respiratory band over time
    power_spectrum = np.abs(Zxx)**2
    resp_power = power_spectrum[resp_band, :].sum(axis=0)
    
    # Check if there are valid peaks
    if resp_power.size > 0 and resp_band.any():
        # Identify the peak in the aggregated power spectrum
        
        peak_idx = np.argmax(resp_power)
        peak_freqs = f[resp_band]
        peak_freq = peak_freqs[np.argmax(power_spectrum[resp_band, peak_idx])]
        rr_bpm = peak_freq * 60
        
        # peak_indices = np.argpartition(resp_power, -3)[-3:]
        # peak_indices = peak_indices[np.argsort(-resp_power[peak_indices])]  # Sort indices by power values
        # peak_freqs = f[resp_band][peak_indices]
        
        # Calculate the mean of the top 3 peak frequencies
        # mean_peak_freq = np.median(peak_freqs)
        
        # Convert frequency to BPM
        # rr_bpm = mean_peak_freq * 60
        
    else:
        rr_bpm = 10  # Default to 0 if no valid peak is found
    
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
        rr = estimate_rr_arp(segment, calculated_fs)
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

