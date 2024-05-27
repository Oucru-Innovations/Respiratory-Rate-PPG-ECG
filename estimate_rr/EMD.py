import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
import pywt
import pandas as pd
from PyEMD import EMD
import sys
sys.path.append('.')
from preprocess.preprocess import preprocess_signal
from scipy.interpolate import interp1d

def estimate_rr_emd(signal, fs):
    
    # Decompose the signal into IMFs using EMD
    emd = EMD()
    imfs = emd(signal)
    
    # Calculate the power spectral density of each IMF
    imf_psds = [np.abs(np.fft.fft(imf))**2 for imf in imfs]
    freqs = np.fft.fftfreq(len(signal), d=1/fs)
    
    # Select the IMF with the dominant frequency in the respiratory range (0.1 to 0.5 Hz)
    best_imf = None
    max_power = 0
    for imf, psd in zip(imfs, imf_psds):
        resp_range = (freqs >= 0.1) & (freqs <= 0.5)
        resp_psd = psd[resp_range]
        if len(resp_psd) > 0:
            max_resp_power = np.max(resp_psd)
            if max_resp_power > max_power:
                max_power = max_resp_power
                best_imf = imf
    
    if best_imf is None:
        return 0  # No relevant IMF found
    
    # Calculate the dominant frequency of the selected IMF
    imf_psd = np.abs(np.fft.fft(best_imf))**2
    dominant_freq = freqs[np.argmax(imf_psd)]
    
    # Convert the frequency to respiratory rate in breaths per minute
    rr_bpm = np.abs(dominant_freq) * 60
    
    return rr_bpm


def get_rr(signal, fs, preprocess=True, signal_type='ECG'):
    if preprocess:
        signal = preprocess_signal(signal, fs,signal_type=signal_type)
    
    rr_bpm = estimate_rr_emd(signal, fs)
    return rr_bpm


# if __name__ == "__main__":

#     # calculated_fs = 256
#     # signal_type = 'ECG'
#     # ecg_data_path = "dataset/public_ecg_data.csv"
#     # ecg_target_path = "dataset/public_ecg_target.csv"
    
#     calculated_fs = 100
#     signal_type = 'PPG'
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
#         rr = get_rr(segment, calculated_fs)
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

