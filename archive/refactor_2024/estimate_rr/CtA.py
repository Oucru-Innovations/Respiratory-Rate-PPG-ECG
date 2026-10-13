import numpy as np
from scipy.signal import find_peaks
from preprocess.preprocess import preprocess_signal

def detect_peaks(signal, fs, min_rr=6, max_rr=30):
    """Detect peaks in a preprocessed PPG signal within an expected respiratory rate range.

    Parameters
    ----------
    signal : array-like
        Input PPG signal.
    fs : int
        Sampling frequency of the signal.
    min_rr : int, default=6
        Minimum expected respiratory rate in breaths per minute.
    max_rr : int, default=30
        Maximum expected respiratory rate in breaths per minute.

    Returns
    -------
    array-like
        Indices of detected peaks.
    """
    preprocessed_signal = preprocess_signal(signal, fs, signal_type='PPG')

    # Calculate peak distance range based on min and max respiratory rates
    min_distance = int(fs * 60 / max_rr)
    max_distance = int(fs * 60 / min_rr)
    
    # Detect peaks with an adaptive threshold
    peaks, _ = find_peaks(preprocessed_signal, distance=min_distance)
    if len(peaks) < 2:
        thresholds = [0.2, 0.1]
        for threshold in thresholds:
            height_threshold = np.mean(preprocessed_signal) + threshold * np.std(preprocessed_signal)
            peaks, _ = find_peaks(preprocessed_signal, distance=min_distance, height=height_threshold)
            if len(peaks) >= 2:
                break

    # Filter out peaks that are too close to each other
    filtered_peaks = [peaks[0]] if len(peaks) > 0 else []
    filtered_peaks.extend(peaks[i] for i in range(1, len(peaks)) if min_distance <= (peaks[i] - peaks[i - 1]) <= max_distance)
    
    return np.array(filtered_peaks)

def calculate_respiratory_rate_from_cta(peaks, fs):
    """Calculate respiratory rate from peak intervals using a peak-based approach.

    Parameters
    ----------
    peaks : array-like
        Indices of detected peaks.
    fs : int
        Sampling frequency of the signal.

    Returns
    -------
    float
        Respiratory rate in breaths per minute.
    """
    if len(peaks) < 2:
        return 0  # Not enough peaks for respiratory rate calculation

    intervals = np.diff(peaks) / fs  # Convert intervals to seconds

    # Filter unrealistic intervals
    min_interval = 60 / 30  # Corresponds to 30 bpm
    max_interval = 60 / 6   # Corresponds to 6 bpm
    valid_intervals = intervals[(intervals >= min_interval) & (intervals <= max_interval)]
    
    if len(valid_intervals) == 0:
        return 0  # No valid intervals

    # Respiratory rate in breaths per minute
    rr_bpm = 60 / np.mean(valid_intervals)
    return rr_bpm

def get_rr(signal, fs, signal_type='PPG', preprocess=True):
    """Calculate respiratory rate from a signal by detecting peaks and calculating intervals.

    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.
    signal_type : str, default='PPG'
        Type of signal, used for preprocessing (e.g., 'PPG' or 'ECG').
    preprocess : bool, default=True
        Whether to preprocess the signal before peak detection.

    Returns
    -------
    float
        Respiratory rate in breaths per minute.
    """
    if preprocess:
        signal = preprocess_signal(signal, fs, signal_type)

    peaks = detect_peaks(signal, fs)
    rr_bpm = calculate_respiratory_rate_from_cta(peaks, fs)
    
    return rr_bpm

# if __name__ == "__main__":
#     calculated_fs = 256
#     signal_type='ECG'
#     ecg_data_path = "dataset/public_ecg_data.csv"
#     ecg_target_path = "dataset/public_ecg_target.csv"
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
