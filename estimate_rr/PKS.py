from scipy.signal import find_peaks
import plotly.graph_objects as go
import numpy as np
from scipy import signal
from preprocess.band_filter import BandpassFilter
from preprocess.preprocess_signal import preprocess_signal


def get_rr(sig, fs, preprocess=True):
    ti = len(sig) / fs
    if preprocess:
        sig = preprocess_signal(sig, fs,filter_type="butterworth", highpass=0.1, lowpass=0.5, degree=1, cutoff=False,
                      cutoff_quantile=0.9)
    local_max = signal.argrelmax(sig)
    thres = np.quantile(sig[local_max], 0.75) * 0.5
    peaks = find_peaks(sig, height=thres)[0]
    # fig = go.Figure()
    # fig.add_traces(go.Scatter())
    peaks_t = np.diff(peaks) * (1 / fs)
    breath_peaks = signal.argrelmax(peaks_t)[0]

    # fig = go.Figure()
    # fig.add_traces(go.Scatter(x=np.arange(len(peaks_t)), y=peaks_t, line=dict(color='blue')))
    # fig.add_traces(go.Scatter(mode='markers', x=breath_peaks, y=peaks_t[breath_peaks],
    #                           marker=dict(color='crimson', size=4)))
    # fig.show()

    return len(breath_peaks) * 60 /ti
