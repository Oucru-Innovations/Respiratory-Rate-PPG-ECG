import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
import pywt
import pandas as pd
from scipy.interpolate import interp1d

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
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

def detect_r_peaks(signal, fs):
    # Find R-peaks using a peak detection algorithm
    distance = int(0.6 * fs)  # Assuming a minimum heart rate of 100 bpm
    peaks, _ = find_peaks(signal, distance=distance, height=np.mean(signal) + 0.5 * np.std(signal))
    return peaks

def interpolate_rr(peaks, fs,num_samples):
    # Calculate the time intervals between R-peaks
    rr_intervals = np.diff(peaks) / fs  # Convert from samples to seconds
    
    # Interpolate the RR intervals to create a smooth curve
    peak_times = peaks[1:] / fs  # Convert from samples to seconds
    interpolator = interp1d(peak_times, rr_intervals, kind='cubic', fill_value='extrapolate')
    
    # Create a time vector for interpolation
    time_vector = np.linspace(peak_times[0], peak_times[-1], num=num_samples)
    
    # Interpolate to get the smooth RR intervals
    interpolated_rr = interpolator(time_vector)
    
    return time_vector, interpolated_rr

def calculate_respiratory_rate(rr_intervals, fs):
    # Interpolate the RR intervals to create a uniformly sampled HRV signal
    rr_times = np.cumsum(rr_intervals) / fs  # Convert to seconds
    hrv_signal = interp1d(rr_times, rr_intervals, kind='cubic', fill_value='extrapolate')

    # Create a time vector for interpolation
    duration = rr_times[-1]
    time_vector = np.linspace(0, duration, int(fs * duration))

    # Get the interpolated HRV signal
    interpolated_hrv = hrv_signal(time_vector)

    # Apply spectral analysis to the HRV signal
    f, Pxx = welch(interpolated_hrv, fs, nperseg=fs*10, scaling='density')

    # Find the dominant frequency within the respiratory range (0.1 to 0.5 Hz)
    resp_range = (f >= 0.1) & (f <= 0.6)
    resp_frequencies = f[resp_range]
    resp_pxx = Pxx[resp_range]
    dominant_frequency = resp_frequencies[np.argmax(resp_pxx)]

    # Convert the dominant frequency to respiratory rate in breaths per minute
    rr_bpm = dominant_frequency * 60

    return rr_bpm


def get_rr(signal, fs, preprocess=True):
    if preprocess:
        signal = preprocess_signal(signal, fs)
    
    # Detect R-peaks
    r_peaks = detect_r_peaks(signal, fs)
    
    # Interpolate RR intervals
    time_vector, interpolated_rr = interpolate_rr(r_peaks, fs,len(signal))
    
    # Calculate the respiratory rate
    rr_bpm = calculate_respiratory_rate(interpolated_rr, fs)
    
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

