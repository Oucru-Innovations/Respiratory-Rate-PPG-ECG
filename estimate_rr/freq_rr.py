from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import resample
import numpy as np


AUTOREGRESSIVE = 0


def get_rr(s, signal_type="PPG", fs=224, dsf=100, method=AUTOREGRESSIVE):
        # resample the time series @ 2Hz
        fs_down = 2
        y = resample(s, dsf)
        # y = interp1(1:numel(thorax), thorax, 1: 1 / 2:20, 'spline');
        y = y - np.mean(y)

        # % Applying the Autoregressive Model method model y using AR order 10
        # a = arburg(y, 10)
        ar_model = AutoReg(y, lags=10).fit()
        ar = ar_model.predict()

        # % obtain the poles of this AR
        r = np.roots(ar)

        print(r)
        filtered_r = [i for i in r if (np.angle(i) >= 10 / 60 * 2 * np.pi / fs_down)]
        filtered_r = [i for i in filtered_r if (np.angle(i) < 25 / 60 * 2 * np.pi / fs_down)]
        print(len(filtered_r))
        # % searching for poles only between 10 Hz to 25 Hz
        # r(angle(r) <= f_low / 60 * 2 * pi / fs_down) = [];
        # r(angle(r) > f_high / 60 * 2 * pi / fs_down) = [];
        # r = sort(r, 'descend');
        # # % plot(r, 'o')
        #
        # # % Determine the respiratory rate
        RR = 60 * np.angle(np.max(filtered_r)) * fs_down / 2 / np.pi

        return RR
