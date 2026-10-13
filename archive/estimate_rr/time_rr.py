from scipy.signal import resample, find_peaks, detrend
from scipy.interpolate import splrep, splev
import numpy as np
from preprocess.band_filter import BandpassFilter



HRV = 0

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
        
def get_rr(s, signal_type="PPG", fs=224, dsf=100, method=HRV):
        hp_cutoff_order = [5, 1]
        lp_cutoff_order = [10, 1]
        # primary_peakdet = 7
        filt = BandpassFilter(band_type='bessel', fs=fs)
        filtered_segment = filt.signal_highpass_filter(s, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
        filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0],
                                                      order=lp_cutoff_order[1])

        sf_ori = fs
        sf = 100
        # sf = 100
        dsf = int(sf / sf_ori * len(filtered_segment))
        ecg = resample(filtered_segment, dsf)
        # ecg = filter_data(ecg, sf, 2, 30, verbose=0)

        # Select only a 20 sec window
        # window = 60
        # start = 1000
        # ecg = ecg[int(start * sf):int((start + window) * sf)]

        # R-R peaks detection
        rr, _ = find_peaks(ecg, distance=40, height=0.5)

        # plt.plot(ecg)
        # plt.plot(rr, ecg[rr], 'o')
        # plt.title('ECG signal')
        # plt.xlabel('Samples')
        # _ =plt.ylabel('Voltage')

        # R-R interval in ms
        rr = (rr / sf) * 1000
        rri = np.diff(rr)
       

        sf_up = 4
        rri_interp = interp_cubic_spline(rri, sf_up)
        hr = 1000 * (60 / rri_interp)
        # print(hr)
        # print('Mean HR: %.2f bpm' % np.mean(hr))

        # Detrend and normalize
        edr = detrend(hr)
        edr = (edr - edr.mean()) / edr.std()

        hp_cutoff_order = [1, 1]
        lp_cutoff_order = [5, 1]
        # primary_peakdet = 7
        filt = BandpassFilter(band_type='bessel', fs=sf)
        filtered_segment = filt.signal_highpass_filter(edr, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
        filtered_segment = filt.signal_lowpass_filter(filtered_segment, cutoff=lp_cutoff_order[0],
                                                      order=lp_cutoff_order[1])

        # Find respiratory peaks
        resp_peaks, _ = find_peaks(filtered_segment, height=0, distance=sf_up)

        # Convert to seconds
        resp_peaks = resp_peaks
        # resp_peaks_diff = np.diff(resp_peaks) / sf_up

        # print(resp_peaks)
        RR = 60 / np.diff(resp_peaks)

        # Plot the EDR waveform
        # plt.plot(filtered_segment, '-')
        # plt.plot(resp_peaks, filtered_segment[resp_peaks], 'o')
        # _ = plt.title('ECG derived respiration')
        # plt.show()

        return np.max(RR)
