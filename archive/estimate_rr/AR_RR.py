import numpy as np
from preprocess.band_filter import BandpassFilter
from preprocess.preprocess_signal import preprocess_signal
from mne.filter import filter_data, resample
from spectrum import arburg
import pandas as pd


# ======================================================================
def get_rr(sig, fs, preprocess=True):
    if preprocess:
        sig = preprocess_signal(sig, fs)
    sf_ori = fs
    sf = 100
    # sf = 100
    dsf = sf / sf_ori
    ecg = resample(sig, dsf)
    ecg = filter_data(ecg, sf, 2, 30, verbose=0)

    # resample the time series @ 2Hz
    fs_down = 2
    y = resample(ecg, dsf)
    # np.interp(ecg)
    # y = interp1(1:numel(thorax), thorax, 1: 1 / 2:20, 'spline')
    y = y - np.mean(y)

    # % Applying the Autoregressive Model method model y using AR order 10
    # a = arburg(y, 10)
    # ar_model = AutoReg(y, lags=10).fit()
    # ar = ar_model.predict()

    # % obtain the poles of this AR
    ar = arburg(y, 10)
    # ar = np.nan_to_num(ar,nan=0)
    # r = np.roots(ar[0])
    r = ar[0]

    # real_part = np.real(r)
    # imaginary_part = np.imag(r)
    # angles = np.angle(r)
    # % searching for poles only between 10 Hz to 25 Hz
    filtered_r = [i for i in r if (np.angle(i) >= 2 * np.pi * (10 / 60)  / fs_down)]
    filtered_r = [i for i in filtered_r if (np.angle(i) < 2 * np.pi * (25 / 60) / fs_down)]

    # % searching for poles only between 10 Hz to 25 Hz
    # r(angle(r) <= f_low / 60 * 2 * pi / fs_down) = []
    # r(angle(r) > f_high / 60 * 2 * pi / fs_down) = []
    # r = sort(r, 'descend');
    # # % plot(r, 'o')
    #
    # # % Determine the respiratory rate
    RR = 60 * np.angle(np.max(filtered_r)) * fs_down / 2 / np.pi

    return RR
