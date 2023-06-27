import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import signal
from skimage import transform
from torchvision.transforms import transforms
from skimage import io
import torch.nn.functional as F
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import resample, detrend
from scipy.signal import butter, filtfilt

class ECGDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets)
        self.fs = 128
        self.nperseg = 256
        self.noverlap = 128
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fs = self.fs
        nperseg = self.nperseg # Number of samples per segment
        noverlap = self.noverlap  # Overlap between segments
        nfft = 2048  # Number of points for FFT
        # Calculate spectrogram
        f, t, Sxx = signal.spectrogram(self.data[index], fs=fs, 
                                       nperseg=nperseg, 
                                       noverlap=noverlap, 
                                       nfft=nfft)
        spectro = 10*np.log10(Sxx+1e-10)
        resized = transform.resize(spectro, spectro.shape)
        normalized = resized/np.max(resized)
        x = torch.tensor(np.stack([normalized] * 3)).float()
        # x = torch.tensor(normalized[np.newaxis, :, :]).float()

        # Resize to 224x224
        x = self.transform(x)
        
        y = torch.tensor(self.targets[0][index])
        return x, y


class ECGClassificationDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.Tensor(data)
        # Convert targets to LongTensor for indexing
        self.targets = torch.LongTensor(targets)
        self.num_classes = 64
        self.fs = 128
        self.nperseg = 256
        self.noverlap = 128
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fs = self.fs
        nperseg = self.nperseg # Number of samples per segment
        noverlap = self.noverlap  # Overlap between segments
        nfft = 2048  # Number of points for FFT
        # Calculate spectrogram
        origin_x = self.data[index]        
        f, t, Sxx = signal.spectrogram(origin_x, fs=fs, 
                                       nperseg=nperseg, 
                                       noverlap=noverlap, 
                                       nfft=nfft)
        spectro = 10*np.log10(Sxx+1e-10)
        resized = transform.resize(spectro, spectro.shape)
        normalized = resized/np.max(resized)
        x = torch.tensor(np.stack([normalized] * 3)).float()
        # x = torch.tensor(normalized[np.newaxis, :, :]).float()

        # Resize to 224x224
        x = self.transform(x)
        
        # y = torch.tensor(self.targets[0][index])
        # Convert target to one-hot encoded vector
        y = F.one_hot(self.targets[0][index], num_classes=self.num_classes).float() 
        return origin_x,x, y

    
class PPGClassificationDataset(Dataset):
    def __init__(self, data, targets,mode='train',fs=100,duration=30):
        self.data = torch.Tensor(data)
        # Convert targets to LongTensor for indexing
        self.targets = torch.LongTensor(targets)
        self.num_classes = 64
        self.mode = mode
        self.fs = fs
        self.duration = duration
        self.nperseg = 256
        self.noverlap = 128
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.data)

    def preprocess_signal(self,sig, lowcut = 0.05 ,highcut = 4.5,fs = 75,order = 2):
        # Calculate nyquist frequency (half of sample rate)
        nyq = 0.5 * fs 

        # Calculate filter coefficients using Butterworth filter design
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')

        # Apply filter to PPG signal using filtfilt to avoid phase distortion
        filtered_ppg = filtfilt(b, a, -sig)
        
        # Enhance dicrotic notches by taking second derivative
        sig_ppg = np.gradient(np.gradient(filtered_ppg))
        # print(sig_ppg)
        return sig_ppg
    
    def get_class_counts(self):
        """
        Returns a dictionary containing the count of samples for each class.
        """
        class_counts = {}
        for i in range(self.num_classes):
            count = (self.targets == i).sum().item()
            class_counts[i] = max(count, 0)  # set count to 0 if it's negative
        return class_counts
    
    def get_class_weights(self):
        """
        Returns a tensor containing the inverse of the class counts as class weights.
        """
        class_counts = self.get_class_counts()
        class_weights = torch.zeros(self.num_classes)
        for c in range(self.num_classes):
            if class_counts[c] > 0:
                class_weights[c] = 1.0 / class_counts[c]
        return class_weights
    
    def __getitem__(self, index):
        fs = self.fs
        nperseg = self.nperseg # Number of samples per segment
        noverlap = self.noverlap  # Overlap between segments
        nfft = 1024  # Number of points for FFT
        # Calculate spectrogram
        origin_x = self.data[index]
        
        # ================== Start add rr to the signal =========================
        # try:
        #     # filter signal
        #     ecg_signal = origin_x.cpu().numpy().reshape(-1)
        #     ecg_signal = signal.detrend(ecg_signal) # remove baseline
        #     b,a = signal.butter(2,[0.1, 4], btype="bandpass",fs=self.fs)
        #     ecg_signal = signal.filtfilt(b,a,ecg_signal)                
        #     #compute RR
        #     # rpeaks,_ = ecg.hamilton_segmenter(ecg_signal,self.fs)
        #     rpeaks = (signal.find_peaks(ecg_signal)[0])
        #     rr_intervals = np.diff(rpeaks)                
        #     #Calculate PSD
        #     f, psd = signal.welch(rr_intervals, fs=self.fs, nperseg= 1024)
        #     f = f/self.fs
        #     #Identify RR frequency component
        #     resp_freq_range = (0.1,0.8)
        #     resp_freq_mask = np.logical_and(f >= resp_freq_range[0], f <= resp_freq_range[1])
        #     max_resp_freq_idx = np.argmax(psd[resp_freq_mask]) 
        #     resp_rate = f[resp_freq_mask][max_resp_freq_idx] * 60
        # except Exception as err:
        #     resp_rate = 20        
        # # Define the parameters
        # breathing_rate = resp_rate # breaths per minute
        # duration = self.duration # seconds
        # sampling_rate = self.fs # Hz
        # # Calculate the number of samples
        # num_samples = duration * sampling_rate
        # # Generate the time axis
        # t = np.linspace(0, duration, num_samples)
        # impedance_signal = signal.resample((1.05 + np.sin(2 * np.pi * breathing_rate * t / 60))/2.1, 
        #                                    len(origin_x))                
        # combine_data = np.multiply(origin_x,impedance_signal)
        # ================== End add rr to the signal =========================

        origin_x = self.preprocess_signal(origin_x,fs=100)
        
        f, t, Sxx = signal.spectrogram(origin_x, fs=fs, 
                                       nperseg=nperseg, 
                                       noverlap=noverlap, 
                                       nfft=nfft)
        spectro = 10*np.log10(Sxx+1e-10)
        resized = transform.resize(spectro, spectro.shape)
        normalized = resized/np.max(resized)
        x = torch.tensor(np.stack([normalized] * 3)).float()
        # x = torch.tensor(normalized[np.newaxis, :, :]).float()

        # Resize to 224x224
        x = self.transform(x)
        
        # display the image using skimage's imshow function
        # img_arr = np.transpose(x.detach().numpy(), (1, 2, 0))
        # io.imshow(img_arr)
        # show the plot window
        # io.show()
        
        if self.mode == 'train':        
            # y = torch.tensor(self.targets[0][index])
            # Convert target to one-hot encoded vector
            y = F.one_hot(self.targets[0][index], num_classes=self.num_classes).float() 
            return origin_x,x, y
        elif self.mode == 'test':
            return origin_x,x
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))