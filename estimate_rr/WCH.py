import numpy as np
from scipy import signal
import scipy
from preprocess.preprocess import preprocess_signal


def get_rr(sig, fs, preprocess=True, downsample_fs=4):
    ti = len(sig)/fs
    if preprocess:
        sig = preprocess_signal(sig, fs)
    # Find the welch periodogram
    segment_length = min(1024, len(sig))  # np.power(2,downsample_fs)
    overlap = int(segment_length / 2)
    f, Pxx = signal.welch(sig, fs, nperseg=1024, noverlap=overlap)
    # print(Pxx)
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=np.arange(len(sig)),y=sig,line=dict(color='blue')))
    # fig.add_trace(go.Scatter(x=f, y=Pxx, line=dict(color='crimson')))
    # fig.show()
    valid_peaks = find_spectral_peak(spectral_power=Pxx, frequency=f)
    # print(f)
    return len(valid_peaks)*60/ti


def find_spectral_peak(spectral_power, frequency):
    # cand_els = []
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=np.arange(len(spectral_power)), y=spectral_power, line=dict(color='crimson')))
    # fig.show()

    spectral_peaks = scipy.signal.argrelmax(spectral_power, order=1)[0]
    # power_dev = spectral_power - np.min(spectral_power)

    valid_signal = np.where((frequency[spectral_peaks] > 0) & (frequency[spectral_peaks] < 2))
    return frequency[spectral_peaks[valid_signal]]
