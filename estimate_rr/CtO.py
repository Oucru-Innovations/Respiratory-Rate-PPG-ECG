import numpy as np
from scipy import signal
from preprocess.band_filter import BandpassFilter
from preprocess.preprocess_signal import preprocess_signal


def get_rr(sig, fs, preprocess=True):
    ti = len(sig) / fs
    # Step 1 preprocess with butterworth filter - 0.1-0.5 -> depend on the device
    if preprocess:
        # pro_sig = preprocess_signal(sig, fs)
        sig = preprocess_signal(sig, fs)
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=np.arange(len(sig)),y=sig,line=dict(color='blue')))
    # fig.add_trace(go.Scatter(x=np.arange(len(pro_sig)), y=pro_sig,line=dict(color='crimson')))
    # fig.show()
    # Step 2
    local_max = signal.argrelmax(sig, order=1)[0]  # if the diff is greater than 2 continuous points
    local_min = signal.argrelmin(sig, order=1)[0]

    # Step 3 define the local max threshold by taking the 0.75 quantile with the confidence 0.95
    max_threshold = np.quantile(sig[local_max], 0.75)*0.95

    # Step 4 find the valid resp cycle
    resp_markers = get_valid_rr(sig, local_min, local_max, max_threshold)
    # print(len(resp_markers))

    return len(resp_markers) * 60 /ti


def get_valid_rr(sig, local_min, local_max, thres):
    # extrema_indices = np.sort(list(local_min) + list(local_max))
    resp_markers = []
    rel_peaks = local_max[sig[local_max] > thres]
    rel_troughs = local_min[sig[local_min] < 0]
    for i in range(len(rel_peaks) - 1):
        cyc_rel_troughs = (np.where((rel_troughs > rel_peaks[i]) &
                                    (rel_troughs < rel_peaks[i + 1])))[0]
        if len(cyc_rel_troughs) == 1:
            resp_markers.append((rel_peaks[i], rel_peaks[i + 1]))
    return resp_markers
