import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from preprocess.preprocess import preprocess_signal

def remove_heartbeat_trend(signal, fs):
    """Remove high-frequency components (e.g., heartbeat) from a signal to isolate respiratory trend.

    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.

    Returns
    -------
    array-like
        Signal with high-frequency components removed.
    """
    b, a = butter(3, 0.5 / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal)

def detect_respiratory_peaks(signal, fs):
    """Detect peaks in a respiratory signal to identify respiratory cycles.

    Parameters
    ----------
    signal : array-like
        Input signal from which respiratory peaks will be detected.
    fs : int
        Sampling frequency of the signal.

    Returns
    -------
    array-like
        Indices of detected respiratory peaks in the signal.
    """
    respiratory_signal = remove_heartbeat_trend(signal, fs)
    min_distance = int(1.5 * fs)  # Minimum distance between peaks (40 breaths per minute max)
    threshold = np.mean(respiratory_signal) - np.std(respiratory_signal)

    peaks, _ = find_peaks(respiratory_signal, distance=min_distance, height=threshold)
    return peaks

def calculate_respiratory_rate(peaks, duration):
    """Calculate respiratory rate in breaths per minute based on detected peaks.

    Parameters
    ----------
    peaks : array-like
        Indices of detected peaks in the signal.
    duration : float
        Duration of the signal in seconds.

    Returns
    -------
    float
        Respiratory rate in breaths per minute.
    """
    if len(peaks) < 2:
        return 0  # Not enough peaks to calculate a respiratory rate

    rr_bpm = (len(peaks) - 1) * 60 / duration
    return rr_bpm

def get_rr(signal, fs, preprocess=True, signal_type='ECG'):
    """Estimate respiratory rate from a signal.

    Parameters
    ----------
    signal : array-like
        Input signal for respiratory rate estimation.
    fs : int
        Sampling frequency of the signal.
    preprocess : bool, default=True
        Whether to preprocess the signal to remove noise and artifacts.
    signal_type : str, default='ECG'
        Type of signal ('ECG' or 'PPG').

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute.
    """
    if preprocess:
        signal = preprocess_signal(signal, fs, signal_type=signal_type)

    respiratory_peaks = detect_respiratory_peaks(signal, fs)
    duration = len(signal) / fs
    return calculate_respiratory_rate(respiratory_peaks, duration)

# if __name__ == "__main__":
#     calculated_fs = 256
#     ecg_data_path = "dataset/public_ecg_data.csv"
#     ecg_target_path = "dataset/public_ecg_target.csv"
    
#     # calculated_fs = 100
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

