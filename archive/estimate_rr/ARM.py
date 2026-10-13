import scipy.signal
import plotly.graph_objects as go
from spectrum import pburg
import numpy as np
from preprocess.preprocess_signal import preprocess_signal


# ======================================================================
def get_rr(sig, fs, upperbound_freq=6, preprocess=True):
    ti = len(sig) / fs
    if preprocess:
        sig = preprocess_signal(sig, fs, filter_type="butterworth", highpass=0.1, lowpass=1.5,
                      degree=5,cutoff=False, cutoff_quantile=0.9,resampling_rate=upperbound_freq)

    # fig = go.Figure()
    # # fig.add_trace(go.Scatter(x=np.arange(len(sig)),y=sig,line=dict(color='blue')))
    # fig.add_trace(go.Scatter(x=np.arange(len(sig)), y=sig, line=dict(color='crimson')))
    # fig.show()

    # apply power spectrum autoregression

    nfft = 1024

    psd_list = []
    for order in range(2, 21):
        # arburg(sig, order,NFFT=nfft, scale_by_freq=True)
        p = pburg(sig, order, NFFT=nfft, scale_by_freq=True)
        psd = p.psd
        psd_list.append(psd)

    p_F = np.arange(0, upperbound_freq, upperbound_freq / (nfft * 0.5))

    spectral_power = np.median(np.array(psd_list), axis=0)
    spectral_peak = find_spectral_peak(p, spectral_power, p_F)


    return 60*spectral_peak/ti


def find_spectral_peak(p, spectral_power, frequency):
    # cand_els = []
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=np.arange(len(spectral_power)), y=spectral_power, line=dict(color='crimson')))
    # fig.show()
    # spectral_peaks = scipy.signal.find_peaks(spectral_power)
    spectral_peaks = scipy.signal.argrelmax(spectral_power, order=1)[0]
    # power_dev = spectral_power - np.min(spectral_power)

    # ar = p.ar
    # angles = np.angle(ar)
    valid_signal = np.where((frequency[spectral_peaks] > 0) & (frequency[spectral_peaks] < 2))
    return len(valid_signal[0])
