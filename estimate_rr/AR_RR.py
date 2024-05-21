import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from statsmodels.tsa.ar_model import AutoReg
from preprocess.preprocess import preprocess_signal
import pandas as pd
import pywt

mother_wavelets = ['db4', 'sym4', 'coif1', 'bior1.3', 'rbio1.3']
# Selected modes to try
selected_modes = ['zero', 'constant', 'symmetric', 'periodic', 
                  'smooth', 'reflect', 'periodization']


def find_resp_peaks(sig, fs, 
                    trapezoid_fs_ratio = 0.7,
                    interval_lb = 0.5, interval_hb = 10, 
                    amplitude_mean_ratio=0.7,slope_lb= 0.1, area_lb=0.5):
       
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

def get_rr(sig, fs, preprocess=True, use_wavelet=True):
    if preprocess:
        sig = preprocess_signal(sig,fs)
    
    # Directly find respiratory peaks in the filtered signal
    rr_bpm = find_resp_peaks(sig, fs)
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

