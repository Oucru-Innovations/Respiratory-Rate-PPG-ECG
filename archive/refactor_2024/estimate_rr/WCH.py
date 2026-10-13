import numpy as np
from scipy.signal import find_peaks, welch, stft
from scipy.fft import fft, fftfreq
from preprocess.preprocess import preprocess_signal

def estimate_rr_fft(sig, fs):
    """Estimate respiratory rate using FFT analysis.

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute.
    """
    freqs = fftfreq(len(sig), 1 / fs)
    fft_spectrum = fft(sig)
    psd = np.abs(fft_spectrum) ** 2

    # Select respiratory frequency range
    resp_range = (freqs >= 0.1) & (freqs <= 0.5)
    resp_psd, resp_freqs = psd[resp_range], freqs[resp_range]
    
    if resp_freqs.size == 0:
        return 0  # No valid respiratory frequency found

    dominant_freq = resp_freqs[np.argmax(resp_psd)]
    return np.abs(dominant_freq) * 60  # Convert to breaths per minute

def estimate_rr_welch(sig, fs):
    """Estimate respiratory rate using Welch's method for power spectral density.

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute.
    """
    freqs, psd = welch(sig, fs, nperseg=len(sig), noverlap=2)

    # Select respiratory frequency range
    resp_range = (freqs >= 0.1) & (freqs <= 0.5)
    resp_psd, resp_freqs = psd[resp_range], freqs[resp_range]

    if resp_freqs.size == 0:
        return 0  # No valid respiratory frequency found

    dominant_freq = resp_freqs[np.argmax(resp_psd)]
    return np.abs(dominant_freq) * 60  # Convert to breaths per minute

def estimate_rr_stft(sig, fs):
    """Estimate respiratory rate using Short-Time Fourier Transform (STFT).

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute.
    """
    f, t, Zxx = stft(sig, fs, nperseg=len(sig), noverlap=2)
    psd = np.abs(Zxx) ** 2

    # Select respiratory frequency range
    resp_range = (f >= 0.1) & (f <= 0.5)
    resp_psd, resp_freqs = psd[resp_range, :], f[resp_range]

    if resp_freqs.size == 0:
        return 0  # No valid respiratory frequency found

    # Find dominant frequency in each time segment and average them
    dominant_freqs = resp_freqs[np.argmax(resp_psd, axis=0)]
    avg_dominant_freq = np.mean(dominant_freqs)
    return np.abs(avg_dominant_freq) * 60  # Convert to breaths per minute

def get_rr(sig, fs, preprocess=True, signal_type='ECG'):
    """Estimate respiratory rate by combining results from FFT, STFT, and Welch methods.

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency.
    preprocess : bool, default=True
        Whether to preprocess the signal.
    signal_type : str, default='ECG'
        Type of signal (ECG or PPG).

    Returns
    -------
    float
        Combined estimated respiratory rate in breaths per minute.
    """
    if preprocess:
        sig = preprocess_signal(sig, fs, signal_type)

    rr_fft = estimate_rr_fft(sig, fs)
    rr_stft = estimate_rr_stft(sig, fs)
    rr_welch = estimate_rr_welch(sig, fs)
    return np.mean([rr_fft, rr_stft, rr_welch])  # Combined average rate

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

