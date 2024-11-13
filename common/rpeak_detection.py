"""R peak detection approaches for PPG and ECG"""

import numpy as np
from sklearn.cluster import KMeans
from scipy import signal
from vitalDSP.physiological_features.waveform import WaveformMorphology

def get_moving_average(q, w):
    """Calculate the moving average of a signal.

    Parameters
    ----------
    q : array-like
        Input signal.
    w : int
        Window size for moving average.

    Returns
    -------
    array-like
        Smoothed signal.
    """
    q_padded = np.pad(q, (w // 2, w - 1 - w // 2), mode='edge')
    return np.convolve(q_padded, np.ones(w) / w, 'valid')

class PeakDetector:
    """Detects R peaks in ECG and PPG signals.

    Parameters
    ----------
    wave_type : str, default='ppg'
        Type of signal ('ecg' or 'ppg').
    fs : int, default=100
        Sampling frequency.
    """

    def __init__(self, wave_type='ppg', fs=100):
        self.wave_type = wave_type
        self.fs = fs

    def ecg_detector(self, s):
        """Detect R peaks in an ECG signal using waveform morphology.

        Parameters
        ----------
        s : array-like
            ECG signal.

        Returns
        -------
        r_peaks : array-like
            Detected R peaks.
        trough_list : array-like
            Detected troughs.
        """
        waveform = WaveformMorphology(waveform=s, fs=self.fs)
        return waveform.r_peaks, waveform.detect_ecg_session()[:, 0]

    def ppg_detector(self, s):
        """Detect systolic peaks in a PPG signal using waveform morphology.

        Parameters
        ----------
        s : array-like
            PPG signal.

        Returns
        -------
        r_peaks : array-like
            Detected systolic peaks.
        trough_list : array-like
            Detected troughs.
        """
        waveform = WaveformMorphology(waveform=s, fs=self.fs)
        return waveform.systolic_peaks, waveform.detect_ppg_session()[:, 0]

    def compute_feature(self, s, local_extrema):
        """Compute amplitude and mean-difference features for local extrema.

        Parameters
        ----------
        s : array-like
            Signal.
        local_extrema : array-like
            Indices of local extrema (peaks or troughs).

        Returns
        -------
        array-like
            Features for clustering.
        """
        amplitude = s[local_extrema]
        mean_diff = np.diff(amplitude, prepend=amplitude[0], append=amplitude[-1])
        return np.column_stack((amplitude, mean_diff))

    def detect_peak_trough_clusterer(self, s):
        """Detect peaks and troughs using clustering with KMeans.

        Parameters
        ----------
        s : array-like
            Signal to be analyzed.

        Returns
        -------
        systolic_peaks_idx : array-like
            Indices of detected systolic peaks.
        trough_idx : array-like
            Indices of detected troughs.
        """
        local_maxima, local_minima = signal.argrelmax(s)[0], signal.argrelmin(s)[0]
        clusterer = KMeans(n_clusters=2, n_init=10)

        systolic_group = clusterer.fit_predict(self.compute_feature(s, local_maxima))
        systolic_peaks_idx = local_maxima[systolic_group == systolic_group.max()]
        trough_group = clusterer.fit_predict(self.compute_feature(s, local_minima))
        trough_idx = local_minima[trough_group == trough_group.min()]

        return systolic_peaks_idx, trough_idx

    def get_ROI(self, s, mva):
        """Identify regions of interest (ROI) in a signal based on the moving average.

        Parameters
        ----------
        s : array-like
            Signal to analyze.
        mva : array-like
            Moving average of the signal.

        Returns
        -------
        tuple of list
            Lists containing the start and end indices of each ROI.
        """
        start_pos = [idx for idx in range(len(s) - 1) if mva[idx] > s[idx] and mva[idx + 1] < s[idx + 1]]
        end_pos = [idx for idx in range(len(s) - 1) if mva[idx] < s[idx] and mva[idx + 1] > s[idx + 1]]
        if len(start_pos) > len(end_pos):
            end_pos.append(len(s) - 1)
        return start_pos, end_pos

    def detect_peak_trough_adaptive_threshold(self, s, adaptive_size=0.75):
        """Detect peaks and troughs using an adaptive threshold approach.

        Parameters
        ----------
        s : array-like
            Signal to analyze.
        adaptive_size : float, default=0.75
            Size of the adaptive window as a fraction of the signal's sampling frequency.

        Returns
        -------
        tuple of list
            Lists of peak and trough indices.
        """
        adaptive_window = int(adaptive_size * self.fs)
        adaptive_threshold = get_moving_average(s, adaptive_window * 2 + 1)

        start_ROIs, end_ROIs = self.get_ROI(s, adaptive_threshold)
        peak_finalist = [np.argmax(s[start:end + 1]) + start for start, end in zip(start_ROIs, end_ROIs)]
        trough_finalist = [np.argmin(s[peak_finalist[i]:peak_finalist[i + 1]]) + peak_finalist[i]
                           for i in range(len(peak_finalist) - 1)]
        return peak_finalist, trough_finalist

    def detect_peak_trough_default_scipy(self, s):
        """Detect peaks and troughs using SciPy's default peak finding.

        Parameters
        ----------
        s : array-like
            Signal to analyze.

        Returns
        -------
        tuple of list
            Lists of peak and trough indices.
        """
        peak_finalist = signal.find_peaks(s)[0]
        trough_finalist = [np.argmin(s[peak_finalist[i]:peak_finalist[i + 1]]) + peak_finalist[i]
                           for i in range(len(peak_finalist) - 1)]
        return peak_finalist, trough_finalist

    def detect_peak_trough_moving_average_threshold(self, s):
        """Detect peaks using a moving average threshold.

        Parameters
        ----------
        s : array-like
            Signal to analyze.

        Returns
        -------
        tuple of list
            Lists of peak and trough indices.
        """
        Z = np.maximum(0, s)
        y = Z ** 2
        w1, w2 = 12, 67
        ma_peak = get_moving_average(y, w1)
        ma_beat = get_moving_average(y, w2)
        thr1 = ma_beat + 0.02 * np.mean(y) + ma_beat
        block_of_interest = ma_peak > thr1
        BOI_idx = np.where(block_of_interest)[0]
        BOI_diff = np.diff(BOI_idx)
        BOI_width_idx = np.where(BOI_diff > 1)[0]

        peak_finalist = [BOI_idx[left] + np.argmax(y[BOI_idx[left]:BOI_idx[right] + 1])
                         for i in range(len(BOI_width_idx))
                         for left, right in zip([BOI_width_idx[i - 1] if i > 0 else 0], [BOI_width_idx[i]])]
        return peak_finalist, []

# Example usage
# -------------
# detector = PeakDetector(wave_type='ecg', fs=250)
# ecg_signal = np.random.randn(1000)
# r_peaks, troughs = detector.ecg_detector(ecg_signal)
# print("R peaks:", r_peaks)
# print("Troughs:", troughs)
