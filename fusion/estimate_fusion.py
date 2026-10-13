import importlib.util
import pandas as pd
import numpy as np
import sys
sys.path.append(".")
from fusion.fuse_rr import combine_estimations
from estimate_rr import AR_RR, ARM, ARP, \
                        CtA, CtO, EMD, \
                        freq_rr, PKS, PZX, WCH

# Define file paths for the uploaded estimation methods
methods = {
    'AR_RR': AR_RR,
    'ARM': ARM,
    'ARP': ARP,
    'CtA': CtA,
    'CtO': CtO,
    'EMD': EMD,
    'freq_rr': freq_rr,
    'PKS': PKS,
    'PZX': PZX,
    'WCH': WCH
}

def estimate_rr_combined(signal, fs, signal_type='ECG', preprocess=True, fuse_method='voting', weights=None):
    results = []
    for name, method in methods.items():  
        rr_bpm = method.get_rr(signal, fs, preprocess, signal_type)
        results.append(rr_bpm)
    
    combined_rr = combine_estimations(results, method=fuse_method, weights=weights)
    return combined_rr


if __name__ == "__main__":

    calculated_fs = 256
    signal_type = 'ECG'
    ecg_data_path = "dataset/public_ecg_data.csv"
    ecg_target_path = "dataset/public_ecg_target.csv"
    
    # calculated_fs = 100
    # signal_type = 'PPG'
    # ecg_data_path = "dataset/public_ppg_data.csv"
    # ecg_target_path = "dataset/public_ppg_target.csv"
    
    # Load the ECG data and target files again, ensuring correct parsing
    ecg_data = pd.read_csv(ecg_data_path, header=None)
    ecg_target = pd.read_csv(ecg_target_path, header=None)

    # Display the shape and first few rows of the data and target files
    ecg_data_shape = ecg_data.shape
    ecg_target_shape = ecg_target.shape

    ecg_data_head = ecg_data.head()
    ecg_target_head = ecg_target.head()
    
    target_rr = ecg_target.values.flatten()

    # Apply the estimate_rr_peaks function to each segment
    estimated_rr = []
    for index, row in ecg_data.iterrows():
        segment = row.values
        rr = estimate_rr_combined(segment, calculated_fs,signal_type)
        estimated_rr.append(rr)

    # Filter out None values from estimated_rr
    valid_estimates = [(est, tgt) for est, tgt in zip(estimated_rr, target_rr) if est is not None]
    estimated_rr_valid, target_rr_valid = zip(*valid_estimates)

    # Convert to numpy arrays for easier comparison
    estimated_rr_valid = np.array(estimated_rr_valid)
    target_rr_valid = np.array(target_rr_valid)

    # Calculate the Mean Absolute Error (MAE) as a simple metric of accuracy
    mae = np.mean(np.abs(estimated_rr_valid - target_rr_valid))
    print(mae)
    print(np.round(estimated_rr_valid[:200]))
    print( target_rr_valid[:200])  # Display MAE and first 10 estimated vs. target values for verification
