import numpy as np
from scipy.signal import argrelmax, argrelmin
from preprocess.preprocess import preprocess_signal

def get_valid_rr(sig, local_min, local_max, threshold=0):
    """Identify valid respiratory rate cycles based on peaks and troughs.

    Parameters
    ----------
    sig : array-like
        Input signal.
    local_min : array-like
        Indices of local minima in the signal.
    local_max : array-like
        Indices of local maxima in the signal.
    threshold : float, default=0
        Minimum amplitude threshold for identifying valid peaks.

    Returns
    -------
    list of tuple
        List of tuples where each tuple contains the start and end indices of valid respiratory cycles.
    """
    resp_markers = []
    rel_peaks = local_max[sig[local_max] > threshold]
    rel_troughs = local_min[sig[local_min] < 0]

    for i in range(len(rel_peaks) - 1):
        cycle_troughs = rel_troughs[(rel_troughs > rel_peaks[i]) & (rel_troughs < rel_peaks[i + 1])]
        if len(cycle_troughs) == 1:  # Valid respiratory cycle if only one trough is present
            resp_markers.append((rel_peaks[i], rel_peaks[i + 1]))
    
    return resp_markers

def get_rr(sig, fs, preprocess=True, signal_type='ECG', last_n_peaks=3):
    """Estimate respiratory rate (RR) based on valid respiratory cycles.

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.
    preprocess : bool, default=True
        Whether to preprocess the signal before calculating RR.
    signal_type : str, default='ECG'
        Type of signal, used for preprocessing if needed.
    last_n_peaks : int, default=3
        Number of recent peaks to use for calculating RR.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute (BPM).
    """
    if preprocess:
        sig = preprocess_signal(sig, fs, signal_type)

    # Detect local maxima and minima
    local_max = argrelmax(sig, order=4)[0]
    local_min = argrelmin(sig, order=4)[0]

    # Identify valid respiratory cycles based on peaks and troughs
    resp_markers = get_valid_rr(sig, local_min, local_max, threshold=0)

    if len(resp_markers) < last_n_peaks:
        return 0  # Not enough cycles to calculate RR

    # Extract the times of the last `last_n_peaks` peaks
    last_peaks = np.array([marker[1] for marker in resp_markers[-last_n_peaks - 1:]])

    # Calculate intervals between consecutive peaks and determine mean interval
    peak_intervals = np.diff(last_peaks) / fs
    mean_interval = np.mean(peak_intervals)

    # Convert mean interval to respiratory rate in BPM
    rr_bpm = 60 / mean_interval
    return rr_bpm

# if __name__ == "__main__":

#     calculated_fs = 256
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

