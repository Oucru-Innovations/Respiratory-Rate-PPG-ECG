import numpy as np
from scipy import signal
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

    # Step 3 define the local max threshold by taking the 0.75 quantile
    # Compute the subsequent local extrema differences
    rel_peaks, rel_troughs = get_valid_rr(sig, local_min, local_max)
    # print(len(resp_markers))
    return len(rel_peaks) * 60 / ti



def get_valid_rr(sig, local_min, local_max):
    extrema_indices = np.sort(list(local_min) + list(local_max))
    # print(extrema_indices)
    # print(sig[extrema_indices])
    extrema_differences = np.abs(np.diff(sig[extrema_indices]))
    # resp_markers = []
    thres = np.quantile(extrema_differences, 0.75)

    # Step 4 find the pair of subsequent extrema
    # find the min difference -> if differ < threshold-> split into 2
    # continue 'till all greater
    removing_extrema = True
    while removing_extrema:
        extrema_differences = np.abs(np.diff(sig[extrema_indices]))
        min_diff = np.min(extrema_differences)
        min_ind_diff = np.argmin(extrema_differences)
        if min_diff < thres:
            extrema_indices = np.delete(extrema_indices, [min_ind_diff, min_ind_diff + 1])
        else:
            removing_extrema = False
    rel_peaks = np.intersect1d(extrema_indices, local_max)
    rel_troughs = np.intersect1d(extrema_indices, local_min)

    return rel_peaks, rel_troughs
