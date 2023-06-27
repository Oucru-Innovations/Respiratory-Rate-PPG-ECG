from scipy.signal import resample
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from scipy import sparse
from RRest.preprocess.band_filter import BandpassFilter
import pandas  as pd
import ast
from sklearn.preprocessing import MinMaxScaler


def synthesize_nonlinear_dynamic_system(duration, trend_frequency=None,
                                        noise_scale=0.05, noise_mean=0.2,
                                        noise_density=0.2,extend_rate=0.2,
                                        resample_rate=2):
    """
    EXPOSE
    :param width:
    :return:
    """
    x1 = 0.15
    x2 = 0.15
    u = 0.5
    beta = 1
    gamma1 = -0.25
    gamma2 = 0.25
    x1_list = [x1]
    x2_list = [x2]

    dt = 0.1
    num_dt = duration/dt

    trough_idx = []

    for t in np.arange(1, num_dt, dt):
        y1 = 0.5 * (np.abs(x1 + 1) - np.abs(x1 - 1))
        y2 = 0.5 * (np.abs(x2 + 1) - np.abs(x2 - 1))
        dx1 = -x1 + (1 + u) * y1 - beta * y2 + gamma1
        dx2 = -x2 + (1 + u) * y2 + beta * y1 + gamma2

        x1 = x1 + dx1 * dt
        x2 = x2 + dx2 * dt

        if np.random.rand()<extend_rate:
            if np.random.rand()<extend_rate:
                num = np.random.randint(6,12)
            else:
                num = np.random.randint(4, 6)
            x2_ = (signal.resample([x2_list[-1],x2],num,signal.windows.hamming(10))[0]).tolist()[:int(num/2)]
            x2_list = x2_list + x2_
        else:
            x2_list.append(x2)
        # x2_list.append(x2)
        x1_list.append(x1)

    trend_t = np.linspace(1, duration, len(x2_list))
    noise_std = (np.max(x2_list) - np.min(x2_list))*noise_scale
    noise_series = np.random.normal(noise_mean, noise_std, len(trend_t))

    noise_sparse = sparse.random(1,len(trend_t),density=noise_density)
    noise_sparse.data[:] = 1
    noise_series = np.multiply(noise_sparse.toarray().reshape(-1), noise_series)

    trend_volts = 2 * np.sin(trend_t / (2*trend_frequency * np.pi)) + noise_series

    trend_list =  trend_volts + x2_list

    # x1_list = signal.normalize(x1_list,[1]*len(x1_list))
    processed_sig = signal.detrend(trend_list)
    bandFilter = BandpassFilter()
    processed_sig = bandFilter.signal_lowpass_filter(processed_sig,4)
    processed_sig = bandFilter.signal_highpass_filter(processed_sig, 0.5)
    processed_sig = signal.resample(processed_sig, int(len(processed_sig) / resample_rate))

    # processed_sig = synthesize_nonlinear_dynamic_system(duration=30, trend_frequency=200,
    #                                                     noise_scale=0.1, noise_mean=0.5,
    #                                                     noise_density=0.3, extend_rate=0.2,
    #                                                     resample_rate=2.5
    #                                                     )
    scaler = MinMaxScaler(
        feature_range=(min(processed_sig), max(processed_sig))
    )
    file_df = pd.read_csv("v2smartcare.csv", warn_bad_lines=True, error_bad_lines=False)
    df = pd.DataFrame()
    pleth_data = []
    file_df['PLETH'].apply(get_flat, flat=pleth_data)
    df['TIMESTAMP_MS'] = np.arange(0, len(pleth_data)) * 10
    df['PLETH'] = pleth_data
    pleth_data = scaler.fit_transform(np.array(pleth_data[:len(processed_sig)]).reshape(-1, 1)).reshape(-1)

    trough_real = signal.find_peaks(-pleth_data,width=15)[0]
    trough_syn = signal.find_peaks(-processed_sig,width=15)[0]




    #=====================================================================
    #               COMBINE SYN AND REAL
    #====================================================================

    ratio_real_syn = 0.2
    ratio_real_syn_idx = (int) (ratio_real_syn*len(trough_syn))
    insertion_idx = np.random.permutation(trough_syn)




    # =====================================================================
    #               END SYN AND REAL
    # ====================================================================
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=np.arange(len(processed_sig)),
            y=pleth_data[:len(processed_sig)],
            mode="lines",
            name="real_data"
        )
    )

    fig.add_trace(go.Scatter(
        x=trough_real,
        y=pleth_data[trough_real],
        marker=dict(color="crimson", size=12),
        mode="markers"
    ))
    # fig.add_traces(go.Scatter(
    #     x=np.arange(len(trend_volts)),
    #     y=trend_volts, mode="lines",
    #     name="trend_t"
    # ))
    fig.add_traces(go.Scatter(
        x=np.arange(len(processed_sig)),
        y=processed_sig, mode="lines",
        name="processed_sig"
    ))
    fig.add_trace(go.Scatter(
        x=trough_syn,
        y=processed_sig[trough_syn],
        marker=dict(color="orange", size=12),
        mode="markers"
    ))

    fig.show()

    return processed_sig

def get_flat(x,flat):
    flat += ast.literal_eval(x)


def plot_syn(processed_sig):
    scaler = MinMaxScaler(
        feature_range=(min(processed_sig), max(processed_sig))
    )
    file_df = pd.read_csv("v2smartcare.csv", warn_bad_lines=True, error_bad_lines=False)
    df = pd.DataFrame()
    pleth_data = []
    file_df['PLETH'].apply(get_flat,flat=pleth_data)
    df['TIMESTAMP_MS'] = np.arange(0,len(pleth_data))*10
    df['PLETH'] = pleth_data
    pleth_data = scaler.fit_transform(np.array(pleth_data[:len(processed_sig)]).reshape(-1, 1)).reshape(-1)


    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
                x=np.arange(len(processed_sig)),
                y=pleth_data[:len(processed_sig)],
                mode="lines",
                name="real_data"
            )
    )
    # fig.add_traces(go.Scatter(
    #     x=np.arange(len(trend_volts)),
    #     y=trend_volts, mode="lines",
    #     name="trend_t"
    # ))
    fig.add_traces(go.Scatter(
        x=np.arange(len(processed_sig)),
        y=processed_sig, mode="lines",
        name="processed_sig"
    ))

    fig.show()

    # peak_detection = signal.argrelmax()


    print("ahihi")
    # local_minima = argrelextrema(np.array(x2_list), np.less)[0]
    # s = np.array(x2_list[local_minima[-2]:local_minima[-1] + 1])
    #
    # rescale_signal = resample(s,width)
    #
    # window = signal.windows.cosine(len(rescale_signal), 0.5)
    # signal_data_tapered = np.array(window) * (rescale_signal - min(rescale_signal))
    #
    # out_scale = MinMaxScaler().fit_transform(
    #     np.array(signal_data_tapered).reshape(-1, 1))
    # return out_scale.reshape(-1)


def combine_synthesize_data():
    processed_sig = synthesize_nonlinear_dynamic_system(duration=30,trend_frequency=200,
                                    noise_scale = 0.1, noise_mean = 0.5,
                                    noise_density = 0.3,extend_rate=0.2,
                                    resample_rate = 2.5
                                    )
    scaler = MinMaxScaler(
        feature_range=(min(processed_sig), max(processed_sig))
    )
    file_df = pd.read_csv("v2smartcare.csv", warn_bad_lines=True, error_bad_lines=False)
    df = pd.DataFrame()
    pleth_data = []
    file_df['PL' \
            'ETH'].apply(get_flat, flat=pleth_data)
    df['TIMESTAMP_MS'] = np.arange(0, len(pleth_data)) * 10
    df['PLETH'] = pleth_data
    pleth_data = scaler.fit_transform(np.array(pleth_data[:len(processed_sig)]).reshape(-1, 1)).reshape(-1)

    peak_real = signal.find_peaks_cwt(-pleth_data)

    return


synthesize_nonlinear_dynamic_system(duration=30,trend_frequency=200,
                                    noise_scale = 0.1, noise_mean = 0.5,
                                    noise_density = 0.3,extend_rate=0.2,
                                    resample_rate = 2.5
                                    )



