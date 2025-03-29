import json
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy import signal
from skimage import transform
from torchvision.transforms import transforms

class WearableDataset(Dataset):
    def __init__(self, data, targets, config_path='dl/config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets).float()
        self.signal_type = self.config["signal_type"]
        self.fs = self.config["fs_ecg"] if self.signal_type == "ecg" else self.config["fs_ppg"]
        self.nperseg = self.config["nperseg"]
        self.noverlap = self.config["noverlap"]
        self.nfft = self.config["nfft"]
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def preprocess_ecg(self, signal_data):
        f, t, Sxx = signal.spectrogram(signal_data, fs=self.fs,
                                       nperseg=self.nperseg,
                                       noverlap=self.noverlap,
                                       nfft=self.nfft)
        return Sxx

    def preprocess_ppg(self, signal_data):
        nyq = 0.5 * self.fs
        b, a = signal.butter(2, [0.05 / nyq, 4.5 / nyq], btype='band')
        filtered_ppg = signal.filtfilt(b, a, signal_data)
        f, t, Sxx = signal.spectrogram(filtered_ppg, fs=self.fs,
                                       nperseg=self.nperseg,
                                       noverlap=self.noverlap,
                                       nfft=self.nfft)
        return Sxx

    def __getitem__(self, index):
        original_signal = self.data[index].numpy()

        if self.signal_type == "ecg":
            spectrogram = self.preprocess_ecg(original_signal)
        elif self.signal_type == "ppg":
            spectrogram = self.preprocess_ppg(original_signal)
        else:
            raise ValueError("Unsupported signal type.")

        spectrogram = 10 * np.log10(spectrogram + 1e-10)
        normalized_spectro = spectrogram / np.max(spectrogram)
        x = torch.tensor(np.stack([normalized_spectro] * 3)).float()
        x = self.transform(x)

        y = self.targets[index]
        return torch.tensor(original_signal).float(), x, y
