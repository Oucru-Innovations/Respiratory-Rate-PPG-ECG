import numpy as np
import os,sys
sys.path.append('.')
from scipy.signal import find_peaks
from statsmodels.tsa.ar_model import AutoReg
from preprocess.preprocess import preprocess_signal

mother_wavelets = ['db4', 'sym4', 'coif1', 'bior1.3', 'rbio1.3']
# Selected modes to try
selected_modes = ['zero', 'constant', 'symmetric', 'periodic', 
                'smooth', 'reflect', 'periodization']

def estimate_rr_ar_1(sig, fs, interval_lb=0.5, interval_hb=10, amplitude_threshold=0.5):
    """Estimate respiratory rate using peak detection in a signal.

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.
    interval_lb : float, default=0.5
        Lower bound for valid peak intervals in seconds.
    interval_hb : float, default=10
        Upper bound for valid peak intervals in seconds.
    amplitude_threshold : float, default=0.5
        Ratio of mean signal amplitude for thresholding peaks.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute, or None if insufficient peaks are found.
    """
    peaks, properties = find_peaks(sig, distance=fs * 0.6, height=np.median(sig) * amplitude_threshold)
    peak_intervals = np.diff(peaks) / fs  # Intervals between peaks in seconds

    valid_peaks = [
        peaks[i] for i in range(1, len(peaks))
        if interval_lb <= peak_intervals[i - 1] <= interval_hb and properties["peak_heights"][i] > np.mean(sig) * amplitude_threshold
    ]
    
    if len(valid_peaks) < 2:
        return None  # Not enough peaks to estimate respiratory rate

    valid_intervals = np.diff(valid_peaks) / fs
    rr_bpm = 60 / np.mean(valid_intervals)  # Breaths per minute
    return rr_bpm

def estimate_rr_ar_2(sig, fs):
    """Estimate respiratory rate using autoregressive (AR) model on the signal.

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute.
    """
    model = AutoReg(sig, lags=2, old_names=False).fit()
    ar_params = model.params

    ar_psd = np.abs(np.fft.fft(ar_params, n=fs)) ** 2  # Power spectral density
    freqs = np.fft.fftfreq(fs)
    resp_freq = freqs[np.argmax(ar_psd)]  # Dominant frequency

    rr_bpm = np.abs(resp_freq) * 60  # Convert frequency to breaths per minute
    return rr_bpm

def get_rr(sig, fs, signal_type='ECG', preprocess=True):
    """Estimate respiratory rate by combining results from multiple AR models.

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.
    signal_type : str, default='ECG'
        Type of signal ('ECG' or 'PPG').
    preprocess : bool, default=True
        Whether to preprocess the signal before estimating respiratory rate.

    Returns
    -------
    float
        Combined estimated respiratory rate in breaths per minute.
    """
    if preprocess:
        sig = preprocess_signal(sig, fs, signal_type)

    rr_bpm_ar_rr1 = estimate_rr_ar_1(sig, fs)
    rr_bpm_ar_rr2 = estimate_rr_ar_2(sig, fs)

    if rr_bpm_ar_rr1 is None and rr_bpm_ar_rr2 is None:
        return 0
    elif rr_bpm_ar_rr1 is None:
        return rr_bpm_ar_rr2
    elif rr_bpm_ar_rr2 is None:
        return rr_bpm_ar_rr1
    else:
        return (rr_bpm_ar_rr1 + rr_bpm_ar_rr2) / 2  # Combined estimate

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

