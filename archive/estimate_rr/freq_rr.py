import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from preprocess.preprocess import preprocess_signal

def calculate_rr_from_ar_poles(sig, fs, order=8, method='max_power'):
    """Estimate respiratory rate based on the poles of an autoregressive (AR) model.

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.
    order : int, default=8
        Order of the AR model.
    method : str, default='max_power'
        Method for selecting the dominant frequency. Options are 'max', 'median', 'mean', and 'max_power'.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute.
    """
    # Fit autoregressive model and retrieve AR coefficients
    ar_model = AutoReg(sig, lags=order).fit()
    ar_params = np.r_[1, -ar_model.params]  # AR coefficients including lag 0

    # Find poles and their angles
    poles = np.roots(ar_params)
    angles = np.angle(poles)

    # Convert angles to frequencies in Hz and then to breaths per minute (BPM)
    frequencies_hz = angles * fs / (2 * np.pi)
    frequencies_bpm = np.abs(frequencies_hz) * 60

    # Filter frequencies within the respiratory range (6 to 40 BPM)
    respiratory_frequencies = frequencies_bpm[(frequencies_bpm > 6) & (frequencies_bpm < 40)]
    if len(respiratory_frequencies) == 0:
        return 0  # No frequencies in the respiratory range

    # Select dominant frequency based on method
    if method == 'max':
        dominant_frequency = np.max(respiratory_frequencies)
    elif method == 'median':
        dominant_frequency = np.median(respiratory_frequencies)
    elif method == 'mean':
        dominant_frequency = np.mean(respiratory_frequencies)
    elif method == 'max_power':
        powers = np.abs(poles) ** 2  # Power of each pole
        dominant_frequency = respiratory_frequencies[np.argmax(powers)]
    else:
        raise ValueError("Invalid method. Choose from 'max', 'median', 'mean', or 'max_power'.")
    
    return dominant_frequency

def get_rr(signal, fs, signal_type='ECG', preprocess=True, order=10):
    """Estimate respiratory rate from a signal.

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
    order : int, default=10
        Order of the AR model used in the calculation.

    Returns
    -------
    float
        Estimated respiratory rate in breaths per minute.
    """
    if preprocess:
        signal = preprocess_signal(signal, fs, signal_type)
    
    return calculate_rr_from_ar_poles(signal, fs, order=order)

# if __name__ == "__main__":

    
#     calculated_fs = 256
#     signal_type='ECG'
#     ecg_data_path = "dataset/public_ecg_data.csv"
#     ecg_target_path = "dataset/public_ecg_target.csv"
    
#     # calculated_fs = 100
#     # signal_type='PPG'
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