import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
import pywt

def preprocess_signal(sig, fs, highpass=0.15, lowpass=0.8, order=4):
    nyquist = 0.5 * fs
    low = highpass / nyquist
    high = lowpass / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    filtered_sig = filtfilt(b, a, sig)
    
    return filtered_sig

def wavelet_denoise(sig, wavelet='db4', level=1):
    coeffs = pywt.wavedec(sig, wavelet, mode='periodization')
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
    denoised_coeffs = coeffs[:]
    denoised_coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in denoised_coeffs[1:]]
    denoised_sig = pywt.waverec(denoised_coeffs, wavelet, mode='periodization')
    return denoised_sig

def find_resp_peaks(sig, fs, 
                    trapezoid_fs_ratio = 0.2,
                    interval_lb = 0.3, interval_hb = 4, 
                    amplitude_mean_ratio=0.2,slope_lb = 0, area_lb=0.1):
    # Detect peaks in the filtered signal
    peaks, properties = find_peaks(sig, distance=fs*0.6, height=np.median(sig)*0.5)  # Minimum distance of 0.5 seconds between peaks, # Lowered the height threshold
    peak_intervals = np.diff(peaks) / fs  # Convert to seconds
    
    # Analyze peak properties to eliminate invalid peaks
    valid_peaks = []
    for i in range(1, len(peaks)):
        interval = peak_intervals[i-1]
        amplitude = properties["peak_heights"][i]
        # Additional criteria: slope and area under the peak
        slope = (sig[peaks[i]] - sig[peaks[i]-1]) / (1 / fs)
        area = np.trapz(sig[peaks[i]-int(fs*trapezoid_fs_ratio):peaks[i]+int(fs*trapezoid_fs_ratio)], dx=1/fs)
        # Add rules for eliminating invalid peaks based on interval and amplitude
        # if 0.5 <= interval <= 5 and amplitude > np.mean(sig) * 0.5:
        if (interval_lb <= interval <= interval_hb) and (amplitude > np.mean(sig) * amplitude_mean_ratio) \
            and (slope > slope_lb) and (area > area_lb):
            valid_peaks.append(peaks[i])
    
    if len(valid_peaks) < 2:
        return None  # Not enough valid peaks to estimate respiratory rate
    
    valid_intervals = np.diff(valid_peaks) / fs  # Convert to seconds
    rr_bpm = 60 / np.mean(valid_intervals)  # Convert to breaths per minute
    return rr_bpm

def estimate_rr_peaks(sig, fs, preprocess=True, use_wavelet=True):
    if preprocess:
        sig = preprocess_signal(sig, fs)
    if use_wavelet:
        sig = wavelet_denoise(sig)
    
    # Directly find respiratory peaks in the filtered signal
    rr_bpm = find_resp_peaks(sig, fs)
    return rr_bpm