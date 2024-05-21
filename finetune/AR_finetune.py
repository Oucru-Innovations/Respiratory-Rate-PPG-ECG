from sklearn.metrics import make_scorer
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV


def mae_score(y_true, y_pred):
    valid_indices = ~np.isnan(y_pred)
    return np.mean(np.abs(y_true[valid_indices] - y_pred[valid_indices]))

mae_scorer = make_scorer(mae_score, greater_is_better=False)

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
import pywt
from itertools import product

class RespirationRateEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, highpass=0.05, lowpass=1.0, order=4,
                 trapezoid_fs_ratio=0.3, interval_lb=0.4, interval_hb=5,
                 amplitude_mean_ratio=0.3, slope_lb=0.1, area_lb=0.2):
        self.highpass = highpass
        self.lowpass = lowpass
        self.order = order
        self.trapezoid_fs_ratio = trapezoid_fs_ratio
        self.interval_lb = interval_lb
        self.interval_hb = interval_hb
        self.amplitude_mean_ratio = amplitude_mean_ratio
        self.slope_lb = slope_lb
        self.area_lb = area_lb
    
    def preprocess_signal(self, sig, fs):
        nyquist = 0.5 * fs
        low = self.highpass / nyquist
        high = self.lowpass / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return filtfilt(b, a, sig)
    
    def find_resp_peaks(self, sig, fs):
        peaks, properties = find_peaks(sig, height=np.mean(sig) * self.amplitude_mean_ratio)
        peak_intervals = np.diff(peaks) / fs
        valid_intervals = [interval for interval in peak_intervals if self.interval_lb <= interval <= self.interval_hb]
        if len(valid_intervals) > 1:
            return 60 / np.mean(valid_intervals)
        return np.nan
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        fs = 250  # Sampling frequency
        estimated_rr = []
        for sig in X:
            sig = self.preprocess_signal(sig, fs)
            rr = self.find_resp_peaks(sig, fs)
            estimated_rr.append(rr)
        return np.array(estimated_rr)



# Define the parameter grid
param_grid = {
    'highpass': [0.05, 0.1],
    'lowpass': [0.7, 1.0],
    'order': [2, 4],
    'trapezoid_fs_ratio': [0.2, 0.3, 0.4],
    'interval_lb': [0.3, 0.4, 0.5],
    'interval_hb': [4, 5, 6],
    'amplitude_mean_ratio': [0.2, 0.3, 0.4],
    'slope_lb': [0.05, 0.1, 0.15],
    'area_lb': [0.1, 0.2, 0.3]
}

if __name__ == "__main__":

    calculated_fs = 256
    ecg_data_path = "dataset/public_ecg_data.csv"
    ecg_target_path = "dataset/public_ecg_target.csv"
    # Load the ECG data and target files again, ensuring correct parsing
    ecg_data = pd.read_csv(ecg_data_path, header=None)
    ecg_target = pd.read_csv(ecg_target_path, header=None)
    # Instantiate the estimator
    estimator = RespirationRateEstimator()

    # Set up the grid search
    grid_search = GridSearchCV(estimator, param_grid, scoring=mae_scorer, cv=3, verbose=3)

    # Prepare the data for grid search
    X = ecg_data.values
    y = ecg_target.values.flatten()

    # Fit the grid search
    grid_search.fit(X, y)

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(best_params, best_score)