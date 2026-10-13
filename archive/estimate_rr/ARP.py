import numpy as np
from scipy.signal import stft, get_window
from statsmodels.tsa.ar_model import AutoReg
from preprocess.preprocess import preprocess_signal

def estimate_rr_arp_1(signal, fs):
    """Estimate respiratory rate using an autoregressive (AR) model with pole frequency analysis.

    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute, or 0 if no valid frequency is found.
    """
    # Fit an AR model and get coefficients
    model = AutoReg(signal, lags=2, old_names=False).fit()
    ar_params = model.params

    # Calculate pole frequencies from AR coefficients
    poles = np.roots(np.concatenate(([1], -ar_params[1:])))
    angles = np.angle(poles)
    frequencies_hz = angles * fs / (2 * np.pi)

    # Filter frequencies within the respiratory range (0.1 to 0.5 Hz)
    resp_frequencies = frequencies_hz[(0.1 <= frequencies_hz) & (frequencies_hz <= 0.5)]
    if resp_frequencies.size == 0:
        return 0  # No valid respiratory frequency found

    # Dominant frequency in breaths per minute
    dominant_frequency = resp_frequencies[np.argmax(np.abs(resp_frequencies))]
    return np.abs(dominant_frequency) * 60

def estimate_rr_arp_2(signal, fs, window_type='hann', lower_bound=0.12, upper_bound=0.6):
    """Estimate respiratory rate using Short-Time Fourier Transform (STFT) and spectral analysis.

    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.
    window_type : str, default='hann'
        Type of window to apply for STFT.
    lower_bound : float, default=0.12
        Lower frequency bound for respiratory range in Hz (approx. 7 bpm).
    upper_bound : float, default=0.6
        Upper frequency bound for respiratory range in Hz (approx. 36 bpm).

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute, or 10 if no valid peak is found.
    """
    nperseg = fs * 25  # Window size for spectral analysis
    noverlap = int(fs / 4)  # Overlap between segments
    window = get_window(window_type, nperseg)

    # Compute STFT and power spectrum
    f, _, Zxx = stft(signal, fs, window=window, nperseg=nperseg, noverlap=noverlap)
    power_spectrum = np.abs(Zxx) ** 2
    resp_band = (f >= lower_bound) & (f <= upper_bound)

    # Aggregate power in the respiratory band
    resp_power = power_spectrum[resp_band, :].sum(axis=0)

    if resp_power.size > 0 and resp_band.any():
        peak_idx = np.argmax(resp_power)
        peak_freqs = f[resp_band]
        peak_freq = peak_freqs[np.argmax(power_spectrum[resp_band, peak_idx])]
        rr_bpm = peak_freq * 60  # Convert frequency to bpm
    else:
        rr_bpm = 10  # Default to 10 if no valid peak is found

    return rr_bpm

def get_rr(signal, fs, signal_type='ECG', preprocess=True):
    """Estimate respiratory rate by combining results from multiple autoregressive models.

    Parameters
    ----------
    signal : array-like
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
        signal = preprocess_signal(signal, fs, signal_type)

    rr_bpm_1 = estimate_rr_arp_1(signal, fs)
    rr_bpm_2 = estimate_rr_arp_2(signal, fs)
    return (rr_bpm_1 + rr_bpm_2) / 2  # Average estimate

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

