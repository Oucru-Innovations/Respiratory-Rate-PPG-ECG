import numpy as np
from preprocess.preprocess import preprocess_signal

def calculate_respiratory_rate_from_zero_crossings(zero_crossings, duration):
    """Calculate respiratory rate from zero-crossings in the signal.

    Parameters
    ----------
    zero_crossings : array-like
        Indices of zero-crossings in the signal.
    duration : float
        Duration of the signal in seconds.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute. Returns 0 if not enough zero-crossings.
    """
    if len(zero_crossings) < 2:
        return 0  # Insufficient zero-crossings for respiratory rate calculation
    return (len(zero_crossings) - 1) * 60 / duration  # Breaths per minute

def detect_zero_crossings(sig):
    """Detect zero-crossings in the signal.

    Parameters
    ----------
    sig : array-like
        Input signal.

    Returns
    -------
    array-like
        Indices of zero-crossings in the signal.
    """
    return np.where(np.diff(np.sign(sig)))[0]

def get_rr(sig, fs, signal_type='ECG', preprocess=True):
    """Estimate respiratory rate based on zero-crossings in the signal.

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.
    signal_type : str, default='ECG'
        Type of signal ('ECG' or 'PPG').
    preprocess : bool, default=True
        Whether to preprocess the signal.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute.
    """
    if preprocess:
        sig = preprocess_signal(sig, fs, signal_type)

    zero_crossings = detect_zero_crossings(sig)
    duration = len(sig) / fs
    return calculate_respiratory_rate_from_zero_crossings(zero_crossings, duration)

def get_sign(index, trough_indices, peak_indices):
    """Determine the sign of an extremum point (trough or peak) in the signal.

    Parameters
    ----------
    index : int
        Index of the extremum point.
    trough_indices : array-like
        Indices of troughs in the signal.
    peak_indices : array-like
        Indices of peaks in the signal.

    Returns
    -------
    int
        -1 if the index corresponds to a trough, 1 if it corresponds to a peak.
    """
    return -1 if index in trough_indices else 1

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
