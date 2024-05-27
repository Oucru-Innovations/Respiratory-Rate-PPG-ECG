import numpy as np
from scipy import signal
import scipy
import numpy as np
import os,sys
sys.path.append('.')
from scipy.signal import butter, filtfilt, find_peaks
from statsmodels.tsa.ar_model import AutoReg
from preprocess.preprocess import preprocess_signal
import pandas as pd
from scipy.signal import butter, filtfilt, welch
import pywt
from scipy.signal import stft


def estimate_rr_fft(sig, fs):
    
    # Compute FFT and power spectral density
    freqs = np.fft.fftfreq(len(sig), 1/fs)
    fft_spectrum = np.fft.fft(sig)
    psd = np.abs(fft_spectrum)**2
    
    # Identify the dominant frequency within the respiratory range (0.1 to 0.5 Hz)
    resp_range = (freqs >= 0.1) & (freqs <= 0.5)
    resp_psd = psd[resp_range]
    resp_freqs = freqs[resp_range]
    
    if len(resp_freqs) == 0:
        return 0  # No valid respiratory frequency found
    
    dominant_freq = resp_freqs[np.argmax(resp_psd)]
    
    # Convert the frequency to respiratory rate in breaths per minute
    rr_bpm = np.abs(dominant_freq) * 60
    
    return rr_bpm

def estimate_rr_welch(sig, fs):
    
    # Compute Welch's power spectral density
    freqs, psd = welch(sig, fs, nperseg=len(sig), noverlap=2)
    
    # Identify the dominant frequency within the respiratory range (0.1 to 0.5 Hz)
    resp_range = (freqs >= 0.1) & (freqs <= 0.5)
    resp_psd = psd[resp_range]
    resp_freqs = freqs[resp_range]
    
    if len(resp_freqs) == 0:
        return 0  # No valid respiratory frequency found
    
    dominant_freq = resp_freqs[np.argmax(resp_psd)]
    
    # Convert the frequency to respiratory rate in breaths per minute
    rr_bpm = np.abs(dominant_freq) * 60
    
    return rr_bpm

def estimate_rr_stft(sig, fs):
    
    # Compute STFT
    f, t, Zxx = stft(sig, fs, nperseg=len(sig), noverlap=2)
    
    # Calculate the power spectral density
    psd = np.abs(Zxx)**2
    
    # Identify the dominant frequency within the respiratory range (0.1 to 0.5 Hz)
    resp_range = (f >= 0.1) & (f <= 0.5)
    resp_psd = psd[resp_range, :]
    resp_freqs = f[resp_range]
    
    if len(resp_freqs) == 0:
        return 0  # No valid respiratory frequency found
    
    # Find the dominant frequency for each time segment and calculate the average
    dominant_freqs = resp_freqs[np.argmax(resp_psd, axis=0)]
    avg_dominant_freq = np.mean(dominant_freqs)
    
    # Convert the frequency to respiratory rate in breaths per minute
    rr_bpm = np.abs(avg_dominant_freq) * 60
    
    return rr_bpm

# def estimate_rr_wavelet(sig, fs, preprocess=True, signal_type='ECG'):
#     if preprocess:
#         sig = preprocess_signal(sig, fs, signal_type)
    
#     # Compute Wavelet Transform
#     scales = np.arange(1, len(sig) // 2)
#     coef, freqs = pywt.cwt(sig, scales, 'cmor', sampling_period=1/fs)
    
#     # Calculate the power spectral density
#     psd = np.abs(coef)**2
    
#     # Identify the dominant frequency within the respiratory range (0.1 to 0.5 Hz)
#     resp_range = (freqs >= 0.1) & (freqs <= 0.5)
#     resp_psd = psd[resp_range, :]
#     resp_freqs = freqs[resp_range]
    
#     if len(resp_freqs) == 0:
#         return 0  # No valid respiratory frequency found
    
#     # Find the dominant frequency for each scale and calculate the average
#     dominant_freqs = resp_freqs[np.argmax(resp_psd, axis=0)]
#     avg_dominant_freq = np.mean(dominant_freqs)
    
#     # Convert the frequency to respiratory rate in breaths per minute
#     rr_bpm = np.abs(avg_dominant_freq) * 60
    
#     return rr_bpm

# def estimate_rr_wavelet(sig, fs):
#     ti = len(sig)/fs
    
#     # Find the welch periodogram
#     segment_length = min(1024, len(sig))  # np.power(2,downsample_fs)
#     overlap = int(segment_length / 2)
#     f, Pxx = signal.welch(sig, fs, nperseg=1024, noverlap=overlap)
#     # print(Pxx)
#     # fig = go.Figure()
#     # fig.add_trace(go.Scatter(x=np.arange(len(sig)),y=sig,line=dict(color='blue')))
#     # fig.add_trace(go.Scatter(x=f, y=Pxx, line=dict(color='crimson')))
#     # fig.show()
#     valid_peaks = find_spectral_peak(spectral_power=Pxx, frequency=f)
#     # print(f)
#     return len(valid_peaks)*60/ti

# def find_spectral_peak(spectral_power, frequency):
#     # cand_els = []
#     # fig = go.Figure()
#     # fig.add_trace(go.Scatter(x=np.arange(len(spectral_power)), y=spectral_power, line=dict(color='crimson')))
#     # fig.show()

#     spectral_peaks = scipy.signal.argrelmax(spectral_power, order=1)[0]
#     # power_dev = spectral_power - np.min(spectral_power)

#     valid_signal = np.where((frequency[spectral_peaks] > 0) & (frequency[spectral_peaks] < 2))
#     return frequency[spectral_peaks[valid_signal]]

def get_rr(sig, fs, preprocess=True, signal_type='ECG'):
    if preprocess:
        sig = preprocess_signal(sig, fs, signal_type)
    
    rr_fft = estimate_rr_fft(sig, fs)
    rr_stft = estimate_rr_stft(sig, fs)
    # rr_wavelet = estimate_rr_wavelet(sig, fs)
    rr_welch = estimate_rr_welch(sig, fs)
    rr_bpm = np.mean([rr_fft, rr_stft, rr_welch])
    
    return rr_bpm


# if __name__ == "__main__":

#     calculated_fs = 256
#     signal_type = 'ECG'
#     ecg_data_path = "dataset/public_ecg_data.csv"
#     ecg_target_path = "dataset/public_ecg_target.csv"
    
#     # calculated_fs = 100
#     # signal_type = 'PPG'
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

