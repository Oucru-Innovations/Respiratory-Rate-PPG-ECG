import numpy as np
from scipy.signal import butter, filtfilt, detrend, resample, find_peaks, windows
import pywt
from preprocess.band_filter import BandpassFilter

def bandpass_filter(signal, fs, lowcut=0.1, highcut=0.5, order=2):
    """Apply a Butterworth bandpass filter to the input signal.

    Parameters
    ----------
    signal : array-like
        The input signal.
    fs : int
        Sampling frequency of the signal.
    lowcut : float, default=0.1
        Lower cutoff frequency.
    highcut : float, default=0.5
        Upper cutoff frequency.
    order : int, default=2
        Order of the Butterworth filter.

    Returns
    -------
    array-like
        Bandpass-filtered signal.
    """
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def wavelet_denoise(signal, wavelet='db4', level=1):
    """Denoise a signal using wavelet transform.

    Parameters
    ----------
    signal : array-like
        The input signal.
    wavelet : str, default='db4'
        Wavelet type.
    level : int, default=1
        Decomposition level for wavelet transform.

    Returns
    -------
    array-like
        Denoised signal.
    """
    coeffs = pywt.wavedec(signal, wavelet, mode='periodization')
    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(denoised_coeffs, wavelet, mode='periodization')

def preprocess_signal(signal, fs, lowcut=0.1, highcut=0.5, signal_type='ECG'):
    """Preprocess a signal by applying bandpass filter and wavelet denoising.

    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : int
        Sampling frequency of the signal.
    lowcut : float, default=0.1
        Lower cutoff frequency.
    highcut : float, default=0.5
        Upper cutoff frequency.
    signal_type : str, default='ECG'
        Type of signal ('ECG' or 'PPG').

    Returns
    -------
    array-like
        Preprocessed signal.
    """
    if signal_type == 'PPG':
        highcut = 0.4  # Adjust for PPG signal
    filtered_signal = bandpass_filter(signal, fs, lowcut, highcut)
    return wavelet_denoise(filtered_signal)

def preprocess_signal_no_wavelet(sig, fs, filter_type="butterworth", highpass=0.1, lowpass=0.5, degree=1, cutoff=False, resampling_rate=None, cutoff_quantile=0.9):
    """Preprocess signal without wavelet denoising using highpass and lowpass filters.

    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : int
        Sampling frequency.
    filter_type : str, default="butterworth"
        Filter type.
    highpass : float, default=0.1
        Highpass filter cutoff frequency.
    lowpass : float, default=0.5
        Lowpass filter cutoff frequency.
    degree : int, default=1
        Filter order.
    cutoff : bool, default=False
        If True, apply cutoff threshold to filtered signal.
    resampling_rate : int, optional
        Resampling rate.
    cutoff_quantile : float, default=0.9
        Quantile for cutoff threshold.

    Returns
    -------
    array-like
        Preprocessed signal.
    """
    filt = BandpassFilter(band_type=filter_type, fs=fs)
    filtered_segment = filt.signal_highpass_filter(sig, highpass, degree)
    filtered_segment = filt.signal_lowpass_filter(filtered_segment, lowpass, degree)

    if cutoff:
        threshold = np.quantile(np.abs(filtered_segment), cutoff_quantile)
        filtered_segment[np.abs(filtered_segment) < threshold] = 0

    if resampling_rate:
        filtered_segment = resample(filtered_segment, int(resampling_rate / fs * len(filtered_segment)))
        filtered_segment = detrend(filtered_segment, overwrite_data=True)
    return filtered_segment

def tapering(signal_data, window=None, shift_min_to_zero=True):
    """Apply a windowing function to taper the edges of the signal.

    Parameters
    ----------
    signal_data : array-like
        Input signal data.
    window : array-like, optional
        Window function (default is Tukey window).
    shift_min_to_zero : bool, default=True
        Shift minimum value of signal to zero.

    Returns
    -------
    array-like
        Tapered signal.
    """
    if shift_min_to_zero:
        signal_data = signal_data - np.min(signal_data)
    if window is None:
        window = windows.tukey(len(signal_data), 0.9)
    return window * signal_data

def smooth(x, window_len=5, window='flat'):
    """Smooth a signal using a window function.

    Parameters
    ----------
    x : array-like
        Input signal.
    window_len : int, default=5
        Length of the smoothing window.
    window : str, default='flat'
        Type of window ('flat', 'hanning', 'hamming', 'bartlett', 'blackman').

    Returns
    -------
    array-like
        Smoothed signal.
    """
    if x.ndim != 1:
        raise ValueError("Smooth only accepts 1D arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be larger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Invalid window type.")

    if window == 'flat':  # Moving average
        w = np.ones(window_len)
    else:
        w = eval(f'np.{window}(window_len)')

    return np.convolve(w / w.sum(), x, mode='same')

def scale_pattern(s, window_size):
    """Scale a small segment to a specified window size.

    Parameters
    ----------
    s : array-like
        Input segment.
    window_size : int
        Target window size.

    Returns
    -------
    array-like
        Scaled segment.
    """
    if len(s) == window_size:
        return np.array(s)
    elif len(s) < window_size:
        span_ratio = window_size / len(s)
        scale_res = [np.mean(s[int(idx / span_ratio)]) for idx in range(window_size)]
    else:
        scale_res = squeeze_template(s, window_size)
    return smooth(scale_res)

def squeeze_template(s, width):
    """Reduce the length of a template signal to a specified width.

    Parameters
    ----------
    s : array-like
        Input signal.
    width : int
        Target width.

    Returns
    -------
    array-like
        Resized signal.
    """
    s = np.array(s)
    total_len = len(s)
    span_unit = 2
    out_res = []

    for i in range(width):
        centroid = (total_len / width) * i
        left, right = int(centroid) - span_unit, int(centroid) + span_unit
        left, right = max(left, 0), min(right, total_len)
        out_res.append(np.mean(s[left:right]))

    return np.array(out_res)
