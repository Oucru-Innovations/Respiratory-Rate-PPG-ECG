import json
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy import signal
from skimage import transform
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class WearableDataset(Dataset):
    def __init__(self, data, targets, config_path='dl/config/config.json', plot_samples=False):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets).float()
        self.signal_type = self.config["signal_type"]
        self.fs = self.config["fs_ecg"] if self.signal_type == "ecg" else self.config["fs_ppg"]
        self.nperseg = self.config["nperseg"]
        self.noverlap = self.config["noverlap"]
        self.nfft = self.config["nfft"]
        self.plot_samples = plot_samples
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Plot a few samples during init if requested
        if self.plot_samples:
            self._plot_sample_spectrograms(num_samples=3)

    def __len__(self):
        return len(self.data)

    def preprocess_ecg(self, signal_data):
        f, t, Sxx = signal.spectrogram(signal_data, fs=self.fs,
                                       nperseg=self.nperseg,
                                       noverlap=self.noverlap,
                                       nfft=self.nfft)
        return f, t, Sxx

    def preprocess_ppg(self, signal_data):
        nyq = 0.5 * self.fs
        b, a = signal.butter(2, [0.05 / nyq, 4.5 / nyq], btype='band')
        filtered_ppg = signal.filtfilt(b, a, signal_data)
        f, t, Sxx = signal.spectrogram(filtered_ppg, fs=self.fs,
                                       nperseg=self.nperseg,
                                       noverlap=self.noverlap,
                                       nfft=self.nfft)
        return f, t, Sxx

    def _plot_sample_spectrograms(self, num_samples=3):
        indices = np.random.choice(len(self.data), num_samples, replace=False)
        for idx in indices:
            original_signal = self.data[idx].numpy()
            target_rr = self.targets[idx].item()
            
            if self.signal_type == "ecg":
                f, t, Sxx = self.preprocess_ecg(original_signal)
            elif self.signal_type == "ppg":
                f, t, Sxx = self.preprocess_ppg(original_signal)
            else:
                raise ValueError("Unsupported signal type.")

            Sxx_log = 10 * np.log10(Sxx + 1e-10)
            
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
            plt.title(f'Spectrogram - RR: {target_rr:.2f} bpm')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Power (dB)')
            plt.ylim(0, 5)  # Focus on 0–5 Hz where RR (0.1–0.83 Hz) lies
            plt.savefig(f'spectrogram_sample_{idx}_rr_{target_rr:.2f}.png')

            plt.close()

    def __getitem__(self, index):
        original_signal = self.data[index].numpy()

        if self.signal_type == "ecg":
            f, t, spectrogram = self.preprocess_ecg(original_signal)
        elif self.signal_type == "ppg":
            f, t, spectrogram = self.preprocess_ppg(original_signal)
        else:
            raise ValueError("Unsupported signal type.")

        spectrogram = 10 * np.log10(spectrogram + 1e-10)
        # normalized_spectro = spectrogram / np.max(spectrogram)
        normalized_spectro = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())  # Min-max instead
        x = torch.tensor(np.stack([normalized_spectro] * 3)).float()
        x = self.transform(x)

        y = self.targets[index]
        return torch.tensor(original_signal).float(), x, y