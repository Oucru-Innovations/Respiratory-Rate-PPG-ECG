import numpy as np
from scipy.signal import find_peaks
from statsmodels.tsa.ar_model import AutoReg
from scipy.fft import fft, fftfreq
from scipy.linalg import toeplitz, solve_toeplitz
from preprocess.preprocess import preprocess_signal

def estimate_rr_arm_1(signal, fs):
    """Estimate respiratory rate using an ARMA model on the signal.

    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute, or None if estimation fails.
    """
    try:
        lags, ma_lags = 2, 8  # AR and MA order for the model
        n = len(signal)

        # Autocorrelation and Toeplitz matrix setup
        r = np.correlate(signal, signal, mode='full')[-n:]
        ar_coeffs = solve_toeplitz((r[:lags], r[:lags]), r[1:lags + 1])
        ma_coeffs = solve_toeplitz((r[:ma_lags], r[:ma_lags]), r[lags:lags + ma_lags])

        # Calculate ARMA power spectral density
        ar_psd = np.abs(fft(ar_coeffs, n=fs)) ** 2
        ma_psd = np.abs(fft(ma_coeffs, n=fs)) ** 2
        arma_psd = ar_psd + ma_psd

        # Select respiratory range frequency (0.1 to 0.5 Hz)
        freqs = fftfreq(fs)
        valid_idx = (freqs >= 0.1) & (freqs <= 0.5)
        if valid_idx.sum() == 0:
            return np.nan

        resp_freq = freqs[valid_idx][np.argmax(arma_psd[valid_idx])]
        return np.abs(resp_freq) * 60  # Breaths per minute

    except (ValueError, np.linalg.LinAlgError, MemoryError):
        return None  # ARMA model fitting failed

def find_resp_peaks(sig, fs, interval_lb=0.5, interval_hb=5, amplitude_ratio=0.9, slope_lb=0, area_lb=0, trapezoid_ratio=0.7):
    """Detect respiratory peaks in a signal based on criteria such as interval and amplitude.

    Parameters
    ----------
    sig : array-like
        Filtered signal.
    fs : int
        Sampling frequency of the signal.
    interval_lb : float, default=0.5
        Minimum interval between peaks in seconds.
    interval_hb : float, default=5
        Maximum interval between peaks in seconds.
    amplitude_ratio : float, default=0.9
        Threshold ratio for peak height relative to the mean signal amplitude.
    slope_lb : float, default=0
        Minimum slope value for a peak to be considered valid.
    area_lb : float, default=0
        Minimum area under the peak to be valid.
    trapezoid_ratio : float, default=0.7
        Ratio of samples on either side of the peak for area calculation.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute, or None if estimation fails.
    """
    peaks, properties = find_peaks(sig, distance=fs * 0.6, height=np.median(sig) * amplitude_ratio)
    peak_intervals = np.diff(peaks) / fs  # Convert to seconds

    valid_peaks = [
        peaks[i] for i in range(1, len(peaks))
        if interval_lb <= peak_intervals[i - 1] <= interval_hb and
           properties["peak_heights"][i] > np.mean(sig) * amplitude_ratio and
           (sig[peaks[i]] - sig[peaks[i] - 1]) / (1 / fs) > slope_lb and
           np.trapz(sig[peaks[i] - int(fs * trapezoid_ratio):peaks[i] + int(fs * trapezoid_ratio)], dx=1 / fs) > area_lb
    ]
    
    if len(valid_peaks) < 2:
        return None  # Not enough valid peaks

    valid_intervals = np.diff(valid_peaks) / fs
    return 60 / np.mean(valid_intervals)  # Convert to breaths per minute

def estimate_rr_arm_2(sig, fs):
    """Estimate respiratory rate using residuals from an autoregressive (AR) model.

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute, or None if insufficient peaks are found.
    """
    model = AutoReg(sig, lags=15).fit()
    residuals = model.resid
    return find_resp_peaks(residuals, fs)

def get_rr(sig, fs, signal_type='ECG', preprocess=True):
    """Estimate respiratory rate by combining results from multiple ARMA models.

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

    rr_bpm_arm_rr1 = estimate_rr_arm_1(sig, fs)
    rr_bpm_arm_rr2 = estimate_rr_arm_2(sig, fs)

    if rr_bpm_arm_rr1 is None and rr_bpm_arm_rr2 is None:
        return 0
    elif rr_bpm_arm_rr1 is None:
        return rr_bpm_arm_rr2
    elif rr_bpm_arm_rr2 is None:
        return rr_bpm_arm_rr1
    else:
        return (rr_bpm_arm_rr1 + rr_bpm_arm_rr2) / 2  # Average estimate

# if __name__ == "__main__":

#     calculated_fs = 256
#     signal_type = 'ECG'
#     ecg_data_path = "dataset/public_ecg_data.csv"
#     ecg_target_path = "dataset/public_ecg_target.csv"
    
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

