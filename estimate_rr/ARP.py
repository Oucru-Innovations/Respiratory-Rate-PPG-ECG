import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, argrelmax, find_peaks, welch
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal.windows import get_window
import pywt
import scipy
from scipy.signal import stft
import sys
sys.path.append('.')
from preprocess.preprocess import preprocess_signal

def estimate_rr_arp_1(signal, fs, preprocess=True, signal_type='ECG'):
    
    # Fit an AR model to the signal
    model = AutoReg(signal, lags=2, old_names=False)
    model_fit = model.fit()
    
    # Get the AR coefficients
    ar_params = model_fit.params
    
    # Calculate the poles of the AR process
    poles = np.roots(np.concatenate(([1], -ar_params[1:])))
    
    # Calculate the frequencies of the poles
    angles = np.angle(poles)
    frequencies_hz = angles * fs / (2 * np.pi)
    
    # Find the dominant frequency within the respiratory range (0.1 to 0.5 Hz)
    resp_range = (frequencies_hz >= 0.1) & (frequencies_hz <= 0.5)
    resp_frequencies = frequencies_hz[resp_range]
    if len(resp_frequencies) == 0:
        return 0  # No valid respiratory frequency found
    
    dominant_frequency = resp_frequencies[np.argmax(np.abs(resp_frequencies))]
    
    # Convert the dominant frequency to respiratory rate in breaths per minute
    rr_bpm = np.abs(dominant_frequency) * 60
    
    return rr_bpm


def estimate_rr_arp_2(sig, fs,
                    window_type='hann', nperseg=1028, noverlap=128,
                    preprocess=True, use_wavelet=True,
                    lower_bound = 0.12,  # Corresponds to about 6 breaths per minute
                    upper_bound = 0.6  # Corresponds to about  breaths per minute
                    ):
    
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

def get_rr(sig, fs, signal_type='ECG',preprocess=True, use_wavelet=True):
    if preprocess:
        sig = preprocess_signal(sig,fs,signal_type)
    
    # Directly find respiratory peaks in the filtered signal
    rr_bpm_1 = estimate_rr_arp_1(sig, fs)
    rr_bpm_2 = estimate_rr_arp_1(sig, fs)
    rr_bpm_combined = (rr_bpm_1 + rr_bpm_2) / 2
    
    return rr_bpm_combined

# if __name__ == "__main__":

#     # calculated_fs = 256
#     # signal_type='ECG'
#     # ecg_data_path = "dataset/public_ecg_data.csv"
#     # ecg_target_path = "dataset/public_ecg_target.csv"
    
#     calculated_fs = 100
#     signal_type='PPG'
#     ecg_data_path = "dataset/public_ppg_data.csv"
#     ecg_target_path = "dataset/public_ppg_target.csv"
    
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
#         rr = get_rr(segment, calculated_fs,signal_type)
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

