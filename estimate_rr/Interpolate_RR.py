from scipy.interpolate import splrep, splev
from scipy.signal import detrend, find_peaks
import numpy as np
from preprocess.band_filter import BandpassFilter
from preprocess.preprocess_signal import preprocess_signal

# Interpolate and compute HR
def interp_cubic_spline(rri, sf_up=4):
    """
    Interpolate R-R intervals using cubic spline.
    Taken from the `hrv` python package by Rhenan Bartels.

    Parameters
    ----------
    rri : np.array
        R-R peak interval (in ms)
    sf_up : float
        Upsampling frequency.

    Returns
    -------
    rri_interp : np.array
        Upsampled/interpolated R-R peak interval array
    """
    rri_time = np.cumsum(rri) / 1000.0
    time_rri = rri_time - rri_time[0]
    time_rri_interp = np.arange(0, time_rri[-1], 1 / float(sf_up))
    tck = splrep(time_rri, rri, s=0)
    rri_interp = splev(time_rri_interp, tck, der=0)
    return rri_interp


def get_rr(sig, fs, preprocess=True):
    ti = len(sig)/fs
    if preprocess:
        sig = preprocess_signal(sig, fs)

    # R-R peaks detection
    rr, _ = find_peaks(sig, distance=20, height=0.5)
    rr = (rr / fs) * 1000
    rri = np.diff(rr)

    sf_up = 4
    rri_interp = interp_cubic_spline(rri, sf_up)
    hr = 1000 * (60 / rri_interp)
    # print(hr)
    # print('Mean HR: %.2f bpm' % np.mean(hr))

    # Detrend and normalize
    edr = detrend(hr)
    edr = (edr - edr.mean()) / edr.std()

    # hp_cutoff_order = [8, 1]
    # lp_cutoff_order = [10, 1]
    # primary_peakdet = 7
    # filt = BandpassFilter(band_type='bessel', fs=fs)
    # filtered_segment = filt.signal_highpass_filter(edr, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
    # filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
    filtered_segment = preprocess_signal(edr, fs, highpass=2, lowpass=0.5)

    # Find respiratory peaks
    resp_peaks, _ = find_peaks(filtered_segment, height=0, distance=sf_up)

    # Convert to seconds
    # resp_peaks = resp_peaks
    # # resp_peaks_diff = np.diff(resp_peaks) / sf_up
    #
    # print(resp_peaks)
    # breath_rate = 60 / np.diff(resp_peaks)
    # print(breath_rate)
    # print(len(breath_rate))
    #
    # # Plot the EDR waveform
    # plt.plot(filtered_segment, '-')
    # plt.plot(resp_peaks, filtered_segment[resp_peaks], 'o')
    # _ = plt.title('ECG derived respiration')
    # plt.show()
    breath_rate = len(resp_peaks) * 60 / ti
    return breath_rate
