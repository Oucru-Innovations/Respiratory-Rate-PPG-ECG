import torch 
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np
import pywt
from layer_pool import DigitCapsules, CapsuleLayer, WaveletTransform, CapsuleNetwork, WaveletLayer
import biosppy.signals.ecg as ecg
from scipy import signal
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
# import sys
# import os
# sys.path.append("..")
# import estimate_rr.CtA as CtA
#=====================================================================
#               Modify Models -Classification Task
#=====================================================================


class RRConvCapsNet(nn.Module):
    def __init__(self, num_classes):
        super(RRConvCapsNet, self).__init__()

        # Load pre-trained EfficientNet
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')

        # Remove the last linear layer from EfficientNet, since we'll add our own Capsule layer
        del self.backbone._fc
        
        # Define Capsule layer
        self.capsule = nn.Sequential(
            nn.BatchNorm1d(1280),
            nn.Linear(1280, 32*6*6),
            nn.ReLU(inplace=True),
            nn.Reshape(-1, 32, 6, 6),
            DigitCapsules(in_capsules=32*6*6, in_capsule_dim=8, num_classes=num_classes, out_capsules=16, out_capsule_dim=16)
        )

        # Define reconstruction network
        self.decoder = nn.Sequential(
            nn.Linear(16*num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3*w*h),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features from EfficientNet backbone
        x = self.backbone.extract_features(x)

        # Apply Capsule layer
        x = x.view(x.size(0), -1)
        x = self.capsule(x)

        # Reconstruction network
        x_recon = self.decoder(x.view(x.size(0), -1))
        x_recon = x_recon.view(x_recon.size(0), 3, w, h)

        return x, x_recon

class RREfficientNetCapsule(nn.Module):
    def __init__(self, num_classes, nperseg=256, noverlap=128, nfft=2048,
                 fs=224, routing_iterations=3, *args, **kwargs) -> None:
        super(RREfficientNetCapsule, self).__init__(*args, **kwargs)
        
        
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.fs = fs
        self.num_classes = num_classes
        
                # Load pre-trained EfficientNet as the base model
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Freeze all layers except batch normalization and last block of the base model
        for name, param in self.base_model.named_parameters():
            if 'bn' not in name and 'blocks.16.' not in name:
                param.requires_grad = False
        
        # Add Capsule Network after feature extraction of EfficientNet
        self.capsule_layer = CapsuleLayer(in_capsules=32, 
                                          in_capsule_dim=7*7*8,
                                          num_classes=num_classes,
                                          out_capsules=16, 
                                          out_capsule_dim=4,
                                          in_channels=3)

        # Add branch for "rr" feature
        self.rr_branch = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )
        
    def compute_RR(self,x):
        x=x[0]
        # convert the PyTorch tensor to a numpy array
        # ecg_data_np = x.cpu().numpy()
        hop_length = self.nperseg-self.noverlap
        rr_list = []
        # loop over each batch in the input tensor
        for batch in x:
            # # convert back to signal from tensor
            # ecg_signal = torch.istft(batch, n_fft=self.nfft, hop_length=hop_length)
            
            # filter signal
            ecg_signal = batch.cpu().numpy()
            ecg_signal = signal.detrend(ecg_signal) # remove baseline
            b,a = signal.butter(2,[0.1, 4], btype="bandpass",fs=self.fs)
            ecg_signal = signal.filtfilt(b,a,ecg_signal)
            
            #compute RR
            # rpeaks,_ = ecg.hamilton_segmenter(ecg_signal,self.fs)
            rpeaks = (signal.find_peaks(ecg_signal)[0])
            rr_intervals = np.diff(rpeaks)
            
            #Calculate PSD
            f, psd = signal.welch(rr_intervals, fs=self.fs, nperseg= 1024)
            f = f/self.fs
            #Identify RR frequency component
            resp_freq_range = (0.1,0.8)
            resp_freq_mask = np.logical_and(f >= resp_freq_range[0], f <= resp_freq_range[1])
            max_resp_freq_idx = np.argmax(psd[resp_freq_mask]) 
            resp_rate = f[resp_freq_mask][max_resp_freq_idx] * 60
            
            rr_list.append(resp_rate)
        
        return torch.tensor(rr_list).unsqueeze(1).to(x.device)  # Return as tensor and unsqueeze to match shape of ECG data

    def forward(self, x):
        rr = self.compute_RR(x)
        x = x[1]
        
        # Pass input through EfficientNet
        features = self.base_model.extract_features(x)
        
        # Flatten EfficientNet output
        features = torch.flatten(features, start_dim=1)
        
        # Process "rr" feature and concatenate with flattened features
        # rr_features = self.rr_branch(rr)
        # features = torch.cat([features, rr], dim=1)
        
        # Pass concatenated features through Capsule Network
        output = self.capsule_layer(features)
        
        return output

# Define the model architecture
class RRClassifier(nn.Module):
    def __init__(self):
        super(RRClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 64)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='linear')

        self.mean = torch.tensor([0.5, 0.5, 0.5])
        self.std = torch.tensor([0.5, 0.5, 0.5])

    def wavelet_scan(self,x):
        # assume ecg_data is a PyTorch tensor containing the ECG signal
        # sample_rate is the sampling rate of the ECG signal
        wavelet_name = 'db4'  # select a wavelet type
        level = 6  # select the level of decomposition

        # convert the PyTorch tensor to a numpy array
        ecg_data_np = x.cpu().numpy()

        # create an empty list to store the padded wavelet coefficients for each batch
        coeffs_all = []

        # loop over each batch in the input tensor
        for batch in ecg_data_np:
            # perform wavelet decomposition on the current batch
            coeffs = pywt.wavedec(batch, wavelet_name, level=level)
            
            # pad the wavelet coefficients with zeros along the last dimension
            max_len = max([c.shape[-1] for c in coeffs])
            coeffs_padded = [np.pad(c, ((0, 0), (0, 0), (0, max_len - c.shape[-1])), mode='constant') for c in coeffs]
            
            # convert the padded wavelet coefficients to PyTorch tensors and append to the list
            coeffs_tensors = [torch.from_numpy(c) for c in coeffs_padded]
            coeffs_all.append(coeffs_tensors)

        # convert the list of coefficient lists to a nested PyTorch tensor
        coeffs_tensor = torch.stack([torch.stack(coeffs) for coeffs in coeffs_all])
    
    def forward(self, x):
        # Normalize input
        x = (x - self.mean.to(x.device)[None, :, None, None]) / \
            self.std.to(x.device)[None, :, None, None]

        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class WaveletCapsuleNetwork(nn.Module):
    def __init__(self, num_classes=10, wavelet_name='haar', mode='symmetric', level=1):
        super(WaveletCapsuleNetwork, self).__init__()
        
        # Define the wavelet transform layer and capsule network layers
        self.wavelet_transform = WaveletTransform(wavelet_name=wavelet_name, mode=mode, level=level)
        self.capsule_network = CapsuleNetwork(num_classes=num_classes)
        
    def forward(self, x):
        # Apply wavelet transform to the input tensor
        x = self.wavelet_transform(x)
        
        # Apply capsule network to the wavelet coefficients
        x = self.capsule_network(x)
        
        return x
  
# Add Wavelet transformation Layer
class RRWaveletClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, 
                 num_orientations=4, kernel_size=3, 
                 stride=1, padding=1, wavelet_type='morlet', 
                 wavelet_params=None, num_wavelet_features=128):
        super(RRWaveletClassifier, self).__init__()

        # Define the convolutional layers for the input spectrogram
        # Define the convolutional layers for the input spectrogram
        input_channels = input_shape[0]
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)

        # Define the wavelet layer
        # self.wavelet_layer = WaveletLayer(32, num_orientations, kernel_size, stride, padding, wavelet_type, wavelet_params)
        self.wavelet_layer = WaveletLayer(num_scales=3, scale_factor=2, wavelet='morlet', freq=0.25)
        # Define the fully connected layers for classification
        self.fc1 = nn.Linear((64 + num_orientations * 2) * (input_shape[1] // 4) * (input_shape[2] // 4), num_wavelet_features)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(num_wavelet_features, num_classes)

        # Define a separate Conv2d layer to reduce the number of channels before applying the wavelet layer
        self.reduce_channels = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x):
        # Apply the convolutional layers to the input spectrogram
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)

        # Reduce the number of channels before applying the wavelet layer
        x = self.reduce_channels(x)

        # Apply the wavelet layer to the input spectrogram
        wavelet_out = self.wavelet_layer(x)

        # Flatten the output of the convolutional and wavelet layers
        x = torch.flatten(x, start_dim=1)
        wavelet_out = torch.flatten(wavelet_out, start_dim=1)

        # Concatenate the outputs of the convolutional and wavelet layers
        x = torch.cat([x, wavelet_out], dim=1)

        # Apply the fully connected layers for classification
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

class RRClassifierUNet(nn.Module):
    def __init__(self,num_classes):
        super(RRClassifierUNet, self).__init__()

        # Encoder part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder part
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(256+128, 128, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128+64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(64+32, 32, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')

        # Output layer
        self.outconv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

        # Normalization and activation layers
        self.batchnorm1 = nn.BatchNorm2d(128+256)
        self.batchnorm2 = nn.BatchNorm2d(64+128)
        self.batchnorm3 = nn.BatchNorm2d(32+64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder part
        x1 = self.relu(self.conv1(x))
        x2 = self.pool1(x1)
        x2 = self.relu(self.conv2(x2))
        x3 = self.pool2(x2)
        x3 = self.relu(self.conv3(x3))
        x4 = self.pool3(x3)

        # Decoder part with skip connections
        x = self.upconv3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.batchnorm1(x)
        x = self.relu(self.conv4(x))

        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.batchnorm2(x)
        x = self.relu(self.conv5(x))

        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.batchnorm3(x)
        x = self.relu(self.conv6(x))

        # Output layer
        x = self.outconv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return x

class RREfficientNetClassifier(nn.Module):
    def __init__(self,num_classes,fs, *args, **kwargs) -> None:
        super(RREfficientNetClassifier, self).__init__(*args, **kwargs)
        self.fs = fs
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # Freeze the weights of the base model
        for param in self.base_model.parameters():
            param.requires_grad = True
               
        self.num_feats = self.base_model._fc.in_features + 1 + 50
        self.fc1 = nn.Linear(self.num_feats, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def compute_RR(self,x):
        # x=x[0]
        # convert the PyTorch tensor to a numpy array
        # ecg_data_np = x.cpu().numpy()
        rr_list = []
        # loop over each batch in the input tensor
        for batch in x:
            # # convert back to signal from tensor
            # ecg_signal = torch.istft(batch, n_fft=self.nfft, hop_length=hop_length)
            try:
                # filter signal
                ecg_signal = batch.cpu().numpy().reshape(-1)
                ecg_signal = signal.detrend(ecg_signal) # remove baseline
                b,a = signal.butter(2,[0.1, 4], btype="bandpass",fs=self.fs)
                ecg_signal = signal.filtfilt(b,a,ecg_signal)
                
                #compute RR
                # rpeaks,_ = ecg.hamilton_segmenter(ecg_signal,self.fs)
                rpeaks = (signal.find_peaks(ecg_signal)[0])
                rr_intervals = np.diff(rpeaks)
                
                #Calculate PSD
                f, psd = signal.welch(rr_intervals, fs=self.fs, nperseg= 1024)
                f = f/self.fs
                #Identify RR frequency component
                resp_freq_range = (0.1,0.8)
                resp_freq_mask = np.logical_and(f >= resp_freq_range[0], f <= resp_freq_range[1])
                max_resp_freq_idx = np.argmax(psd[resp_freq_mask]) 
                resp_rate = f[resp_freq_mask][max_resp_freq_idx] * 60
            except Exception as err:
                resp_rate = 20
            
            rr_list.append(resp_rate)
        
        return torch.tensor(rr_list).unsqueeze(1).to(x.device)  # Return as tensor and unsqueeze to match shape of ECG data

   
    def wavelet_scan(self,x):  
        # x = x[1]      
        # assume ecg_data is a PyTorch tensor containing the ECG signal
        # sample_rate is the sampling rate of the ECG signal
        wavelet_name = 'db4'  # select a wavelet type
        level = 4  # select the level of decomposition

        # convert the PyTorch tensor to a numpy array
        ecg_data_np = x.cpu().numpy()

        # create an empty list to store the padded wavelet coefficients for each batch
        coeffs_all = []

        # loop over each batch in the input tensor
        for batch in ecg_data_np:
            # perform wavelet decomposition on the current batch
            coeffs = pywt.wavedec(batch, wavelet_name, level=level)
            
            # pad the wavelet coefficients with zeros along the last dimension
            # max_len = max([c.shape[-1] for c in coeffs])
            # coeffs_padded = [np.pad(c, ((0, 0), (0, 0), (0, max_len - c.shape[-1])), mode='constant') 
            #                  for c in coeffs]
            coeffs_padded = coeffs
            
            # convert the padded wavelet coefficients to PyTorch tensors and append to the list
            coeffs_tensors = [torch.from_numpy(c) for c in coeffs_padded[:]]
            coeffs_all.append(coeffs_tensors)

        # convert the list of coefficient lists to a nested PyTorch tensor
        coeffs_tensor = torch.stack([torch.stack(coeffs[:2]) for coeffs in coeffs_all])
        
        
        # Case wavelet on signal
        batch_size, levels, channels, w = coeffs_tensor.shape
        
        level_1_coeffs = coeffs_tensor[:,0,:,:].clone()
        level_1_coeffs = level_1_coeffs.view(batch_size,channels,w)
        # return level_1_coeffs

        ca_coeffs_tensor = coeffs_tensor[:,:,0,:].clone()
        ca_coeffs_tensor = ca_coeffs_tensor.view(batch_size,levels*w)
        
        # ca_coeffs_tensor = ca_coeffs_tensor.unsqueeze(-1)  # Add dummy spatial dimension
        # ca_coeffs_tensor = F.interpolate(ca_coeffs_tensor,size=(20,), mode='linear')
        # ca_interpolated_tensor = torch.mean(ca_coeffs_tensor, dim=1)  # Reduce channels to 1
        
        # Shape is [batch_size, level of decomposition,number of coefficient arrays at each level of decomposition, height dimension of each spectrogram image,idth dimension of each spectrogram image]

        # Decomposes the signal into multiple frequency bands (approximations and details) at different resolution levels.

        # This dimension represents the number of coefficient arrays at each level of decomposition. Specifically, it corresponds to the approximation coefficients (cA) and the two detail coefficients (cD) for each level of decomposition. For example, at level 1 there are 3 coefficient arrays: cA1, cD1 (horizontal), and cD1 (vertical). The actual number of coefficient arrays may vary depending on the wavelet family used and the level of decomposition.
        
        # length of the wavelet coefficients at each level of decomposition..
        # torch.nn.functional.normalize(ca_interpolated_tensor, dim=-1)
        # Compute mean and standard deviation along data dimension
        ca_interpolated_tensor = F.interpolate(ca_coeffs_tensor.unsqueeze(1), 
                                               size=(50,), 
                                               mode='linear').squeeze(1)
        
        normalized_tensor = torch.nn.functional.normalize(ca_interpolated_tensor, dim=-1)
        return normalized_tensor.to(x.device) # shape is [batch_size,]
    
    def forward(self, x, heatmap_layer=None, layer_idx=None):
        origin_x = x[0]
        rr = self.compute_RR(origin_x)
        # ca_interpolated_tensor = self.wavelet_scan(origin_x)
        x = x[1]        
        
        features = self.base_model.extract_features(x)
        if heatmap_layer is not None:
            assert 0 <= heatmap_layer <= len(features) - 1, f"Heatmap layer should be between 0 and {len(features) - 1}"
            heatmap = F.relu(features[heatmap_layer])
            heatmap = F.adaptive_avg_pool2d(heatmap, 1)
            heatmap = heatmap.squeeze()
            # Reshape the heatmap tensor into a 2D image
            # h, w = features.shape[-2:]
            # heatmap = heatmap.reshape((32, 40))
            heatmap = heatmap.detach().cpu().numpy()
        else:
            heatmap = None

        if layer_idx is not None:
            assert 0 <= layer_idx <= len(features) - 1, f"Layer index should be between 0 and {len(features) - 1}"
            feature_map = features[layer_idx]
            # Reshape feature map tensor to 2D image format
            h, w = feature_map.shape[-2:]
            h = 640
            w = 98
            feature_map = feature_map.detach().cpu().numpy().reshape(h, w)
        else:
            feature_map = None
        
        # rr = self.compute_CtA(x)
        # ca_coeffs_tensor,cD_coeffs_tensor = self.wavelet_scan(origin_x)
        # combine features
        # x = self.base_model._avg_pooling(features)
        # x = x.flatten(start_dim=1)
        
        x_features = self.base_model._avg_pooling(features)
        x_features = x_features.flatten(start_dim=1)
        x = torch.cat([x_features, rr.float()], dim=1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if (layer_idx is not None) or (heatmap_layer is not None):
           
            return x, feature_map, heatmap
        return x

class RREfficientNetGMClassifierPPG(nn.Module):
    def __init__(self,num_classes,fs,num_mixtures=5, *args, **kwargs) -> None:
        super(RREfficientNetGMClassifierPPG, self).__init__(*args, **kwargs)
        self.fs = fs
        self.num_mixtures = num_mixtures
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # Freeze the weights of the base model
        for param in self.base_model.parameters():
            param.requires_grad = True
        
        self.num_classes = num_classes
        self.num_feats = self.base_model._fc.in_features
        
        
        # Define trainable parameters for the Gaussian mixture model
        self.mus = nn.Parameter(torch.randn(num_mixtures, self.num_feats))
        self.log_sigmas = nn.Parameter(torch.randn(num_mixtures, self.num_feats))
        self.log_weights = nn.Parameter(torch.randn(num_mixtures))

        # Define the classification layer on top of the Gaussian mixture model
        self.classification_layer = nn.Linear(num_mixtures * num_classes, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.num_feats, num_classes)
        
    def forward(self, x, heatmap_layer=None, layer_idx=None):
        origin_x = x[0]
        x = x[1]        
        
        features = self.base_model.extract_features(x)
        if heatmap_layer is not None:
            assert 0 <= heatmap_layer <= len(features) - 1, f"Heatmap layer should be between 0 and {len(features) - 1}"
            heatmap = F.relu(features[heatmap_layer])
            heatmap = F.adaptive_avg_pool2d(heatmap, 1)
            heatmap = heatmap.squeeze()
            # Reshape the heatmap tensor into a 2D image
            # h, w = features.shape[-2:]
            # heatmap = heatmap.reshape((32, 40))
            heatmap = heatmap.detach().cpu().numpy()
        else:
            heatmap = None

        if layer_idx is not None:
            assert 0 <= layer_idx <= len(features) - 1, f"Layer index should be between 0 and {len(features) - 1}"
            feature_map = features[layer_idx]
            # Reshape feature map tensor to 2D image format
            h, w = feature_map.shape[-2:]
            h = 640
            w = 98
            feature_map = feature_map.detach().cpu().numpy().reshape(h, w)
        else:
            feature_map = None
        
        x_features = self.base_model._avg_pooling(features)
        x_features = x_features.flatten(start_dim=1)       
        
        # Compute the probabilities for each Gaussian mixture component
        normal_dists = torch.distributions.Normal(self.mus, torch.exp(self.log_sigmas))
        log_probs = normal_dists.log_prob(x_features.unsqueeze(1)).sum(dim=-1) + self.log_weights
        probs = torch.softmax(log_probs, dim=1)

        # Compute the Gaussian mixture representation of the input
        gaussian_mixture = (probs.unsqueeze(-1) * self.mus.unsqueeze(0)).sum(dim=1)
        batches = gaussian_mixture.shape[0]
        # Reshape the Gaussian mixture representation to match the expected shape of the classification layer
        
        gaussian_mixture = torch.reshape(gaussian_mixture,
                                         (batches, -1, self.num_mixtures * self.num_classes))
        
        # gaussian_mixture = torch.reshape(gaussian_mixture, 
        #                                  (-1, self.num_mixtures * self.num_classes))

        
        # Pass the Gaussian mixture representation through the classification layer
        weight_transpose = self.classification_layer.weight.transpose(0, 1)
        x = torch.matmul(gaussian_mixture, weight_transpose) + self.classification_layer.bias
        
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x =  x_features.flatten(start_dim=1)
        x = self.fc2(x)
        
        if (layer_idx is not None) or (heatmap_layer is not None):
            return x, feature_map, heatmap
        return x

class RREfficientNetClassifierPPG(nn.Module):
    def __init__(self,num_classes,fs, *args, **kwargs) -> None:
        super(RREfficientNetClassifierPPG, self).__init__(*args, **kwargs)
        self.fs = fs
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # Freeze the weights of the base model
        for param in self.base_model.parameters():
            param.requires_grad = True
               
        self.num_feats = self.base_model._fc.in_features + 1 + 50
        self.fc1 = nn.Linear(self.num_feats, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def compute_RR(self,x):
        # x=x[0]
        # convert the PyTorch tensor to a numpy array
        # ecg_data_np = x.cpu().numpy()
        rr_list = []
        # loop over each batch in the input tensor
        for batch in x:
            # # convert back to signal from tensor
            # ecg_signal = torch.istft(batch, n_fft=self.nfft, hop_length=hop_length)
            try:
                # filter signal
                ecg_signal = batch.cpu().numpy().reshape(-1)
                ecg_signal = signal.detrend(ecg_signal) # remove baseline
                b,a = signal.butter(2,[0.1, 4], btype="bandpass",fs=self.fs)
                ecg_signal = signal.filtfilt(b,a,ecg_signal)
                
                #compute RR
                # rpeaks,_ = ecg.hamilton_segmenter(ecg_signal,self.fs)
                rpeaks = (signal.find_peaks(ecg_signal)[0])
                rr_intervals = np.diff(rpeaks)
                
                #Calculate PSD
                f, psd = signal.welch(rr_intervals, fs=self.fs, nperseg= 1024)
                f = f/self.fs
                #Identify RR frequency component
                resp_freq_range = (0.1,0.8)
                resp_freq_mask = np.logical_and(f >= resp_freq_range[0], f <= resp_freq_range[1])
                max_resp_freq_idx = np.argmax(psd[resp_freq_mask]) 
                resp_rate = f[resp_freq_mask][max_resp_freq_idx] * 60
            except Exception as err:
                resp_rate = 20
            
            rr_list.append(resp_rate)
        
        return torch.tensor(rr_list).unsqueeze(1).to(x.device)  # Return as tensor and unsqueeze to match shape of ECG data

   
    def wavelet_scan(self,x):  
        # x = x[1]      
        # assume ecg_data is a PyTorch tensor containing the ECG signal
        # sample_rate is the sampling rate of the ECG signal
        wavelet_name = 'db4'  # select a wavelet type
        level = 4  # select the level of decomposition

        # convert the PyTorch tensor to a numpy array
        ecg_data_np = x.cpu().numpy()

        # create an empty list to store the padded wavelet coefficients for each batch
        coeffs_all = []

        # loop over each batch in the input tensor
        for batch in ecg_data_np:
            # perform wavelet decomposition on the current batch
            coeffs = pywt.wavedec(batch, wavelet_name, level=level)
            
            # pad the wavelet coefficients with zeros along the last dimension
            # max_len = max([c.shape[-1] for c in coeffs])
            # coeffs_padded = [np.pad(c, ((0, 0), (0, 0), (0, max_len - c.shape[-1])), mode='constant') 
            #                  for c in coeffs]
            coeffs_padded = coeffs
            
            # convert the padded wavelet coefficients to PyTorch tensors and append to the list
            coeffs_tensors = [torch.from_numpy(c) for c in coeffs_padded[:]]
            coeffs_all.append(coeffs_tensors)

        # convert the list of coefficient lists to a nested PyTorch tensor
        coeffs_tensor = torch.stack([torch.stack(coeffs[:2]) for coeffs in coeffs_all])
        # coeffs_tensor
        # Case wavelet on images
        # batch_size, levels, channels, w,wavelength = coeffs_tensor.shape
        
        # level_1_coeffs = coeffs_tensor[:,0,:,:,:].clone()
        # level_1_coeffs = level_1_coeffs.view(batch_size,channels,w,wavelength)
        # # return level_1_coeffs

        # ca_coeffs_tensor = coeffs_tensor[:,:,0,:,:].clone()
        # ca_coeffs_tensor = ca_coeffs_tensor.view(batch_size,levels*w* wavelength)
        
        # cD_coeffs_tensor = coeffs_tensor[:,:,1,:,:].clone()
        # cD_coeffs_tensor = cD_coeffs_tensor.view(batch_size,levels*w* wavelength)
        
        # ca_coeffs_tensor = ca_coeffs_tensor.unsqueeze(-1)  # Add dummy spatial dimension
        # ca_coeffs_tensor = F.interpolate(ca_coeffs_tensor,size=(100,), mode='linear')
        # ca_interpolated_tensor = torch.mean(ca_coeffs_tensor, dim=1)  # Reduce channels to 1
        
        # cD_coeffs_tensor = cD_coeffs_tensor.unsqueeze(-1)  # Add dummy spatial dimension
        # cD_coeffs_tensor = F.interpolate(cD_coeffs_tensor,size=(100,), mode='linear')
        # cD_interpolated_tensor = torch.mean(cD_coeffs_tensor, dim=1)  # Reduce channels to 1

        
        # # Shape is [batch_size, level of decomposition,number of coefficient arrays at each level of decomposition, height dimension of each spectrogram image,idth dimension of each spectrogram image]

        # # Decomposes the signal into multiple frequency bands (approximations and details) at different resolution levels.

        # # This dimension represents the number of coefficient arrays at each level of decomposition. Specifically, it corresponds to the approximation coefficients (cA) and the two detail coefficients (cD) for each level of decomposition. For example, at level 1 there are 3 coefficient arrays: cA1, cD1 (horizontal), and cD1 (vertical). The actual number of coefficient arrays may vary depending on the wavelet family used and the level of decomposition.
        
        # # length of the wavelet coefficients at each level of decomposition..

        # return ca_interpolated_tensor,cD_interpolated_tensor # shape is [batch_size,]
        
        # Case wavelet on signal
        batch_size, levels, channels, w = coeffs_tensor.shape
        
        level_1_coeffs = coeffs_tensor[:,0,:,:].clone()
        level_1_coeffs = level_1_coeffs.view(batch_size,channels,w)
        # return level_1_coeffs

        ca_coeffs_tensor = coeffs_tensor[:,:,0,:].clone()
        ca_coeffs_tensor = ca_coeffs_tensor.view(batch_size,levels*w)
        
        # ca_coeffs_tensor = ca_coeffs_tensor.unsqueeze(-1)  # Add dummy spatial dimension
        # ca_coeffs_tensor = F.interpolate(ca_coeffs_tensor,size=(20,), mode='linear')
        # ca_interpolated_tensor = torch.mean(ca_coeffs_tensor, dim=1)  # Reduce channels to 1
        
        # Shape is [batch_size, level of decomposition,number of coefficient arrays at each level of decomposition, height dimension of each spectrogram image,idth dimension of each spectrogram image]

        # Decomposes the signal into multiple frequency bands (approximations and details) at different resolution levels.

        # This dimension represents the number of coefficient arrays at each level of decomposition. Specifically, it corresponds to the approximation coefficients (cA) and the two detail coefficients (cD) for each level of decomposition. For example, at level 1 there are 3 coefficient arrays: cA1, cD1 (horizontal), and cD1 (vertical). The actual number of coefficient arrays may vary depending on the wavelet family used and the level of decomposition.
        
        # length of the wavelet coefficients at each level of decomposition..
        # torch.nn.functional.normalize(ca_interpolated_tensor, dim=-1)
        # Compute mean and standard deviation along data dimension
        ca_interpolated_tensor = F.interpolate(ca_coeffs_tensor.unsqueeze(1), 
                                               size=(50,), 
                                               mode='linear').squeeze(1)
        
        normalized_tensor = torch.nn.functional.normalize(ca_interpolated_tensor, dim=-1)
        return normalized_tensor.to(x.device) # shape is [batch_size,]
    
    def forward(self, x, heatmap_layer=None, layer_idx=None):
        origin_x = x[0]
        rr = self.compute_RR(origin_x)
        # ca_interpolated_tensor = self.wavelet_scan(origin_x)
        combine_data = x[0]
        ca_interpolated_tensor = self.wavelet_scan(combine_data)
        x = x[1]        
        
        
        # for batch in combine_data:
        #     batch = batch.cpu().numpy().reshape(-1)
        #     fig = go.Figure()
        #     # fig.add_trace(go.Scatter(x=np.arange(len(df_sc)),y=-(df_sc['PLETH']-33175)))
        #     fig.add_trace(go.Scatter(x=np.arange(len(batch)),y=batch))
        #     fig.show()
        
        
        features = self.base_model.extract_features(x)
        if heatmap_layer is not None:
            assert 0 <= heatmap_layer <= len(features) - 1, f"Heatmap layer should be between 0 and {len(features) - 1}"
            heatmap = F.relu(features[heatmap_layer])
            heatmap = F.adaptive_avg_pool2d(heatmap, 1)
            heatmap = heatmap.squeeze()
            # Reshape the heatmap tensor into a 2D image
            # h, w = features.shape[-2:]
            # heatmap = heatmap.reshape((32, 40))
            heatmap = heatmap.detach().cpu().numpy()
        else:
            heatmap = None

        if layer_idx is not None:
            assert 0 <= layer_idx <= len(features) - 1, f"Layer index should be between 0 and {len(features) - 1}"
            feature_map = features[layer_idx]
            # Reshape feature map tensor to 2D image format
            h, w = feature_map.shape[-2:]
            h = 640
            w = 98
            feature_map = feature_map.detach().cpu().numpy().reshape(h, w)
        else:
            feature_map = None
        
        # rr = self.compute_CtA(x)
        # ca_coeffs_tensor,cD_coeffs_tensor = self.wavelet_scan(origin_x)
        # combine features
        # x = self.base_model._avg_pooling(features)
        # x = x.flatten(start_dim=1)
        
        x_features = self.base_model._avg_pooling(features)
        x_features = x_features.flatten(start_dim=1)
        x = torch.cat([x_features, rr.float(), ca_interpolated_tensor.float()], dim=1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if (layer_idx is not None) or (heatmap_layer is not None):
           
            return x, feature_map, heatmap
        return x

class RRConvGRUCapsModel(nn.Module):
    def __init__(self, num_classes):
        super(RRConvGRUCapsModel, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # GRU Layer
        self.gru = nn.GRU(input_size=64*56, hidden_size=64, num_layers=2, batch_first=True)
        
        # Capsule Layer
        self.primary_capsules = PrimaryCapsules(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, num_capsules=8, capsule_dim=8)
        self.digit_capsules = DigitCapsules(num_routes=128, in_channels=16, out_channels=num_classes, capsule_dim=16)
        
    def forward(self, x):
        # Convolutional Layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # Reshape for GRU
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels*height, width).permute(0, 2, 1)
        
        # GRU Layer
        x, _ = self.gru(x)
        
        # Capsule Layer
        x = x.permute(0, 2, 1)  # (batch_size, hidden_size, time_steps)
        x = self.primary_capsules(x)  # (batch_size, num_capsules=8*25, capsule_dim=8)
        x = self.digit_capsules(x)  # (batch_size, num_classes, capsule_dim=16)
        out = x.norm(dim=-1)  # (batch_size, num_classes)
        return out
    
class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_capsules, capsule_dim):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv1d(in_channels, out_channels * num_capsules, kernel_size, stride, padding)
        
    def forward(self, x):
        x = self.conv(x)
        batch_size, _, length = x.size()
        x = x.view(batch_size, self.num_capsules, self.capsule_dim, length).transpose(1,3)
        x = squash(x)
        return x

class DigitCapsules(nn.Module):
    def __init__(self, num_routes, in_channels, out_channels, capsule_dim):
        super(DigitCapsules, self).__init__()
        self.num_routes = num_routes
        self.out_channels = out_channels
        self.capsule_dim = capsule_dim
        self.route_weights = nn.Parameter(torch.randn(1, num_routes, out_channels, capsule_dim, in_channels))
        
    def forward(self, x):
        u_hat = torch.matmul(x[:, None, :, :], self.route_weights)
        b_ij = torch.zeros(1, self.num_routes, self.out_channels, 1, 1)
        iterations = 3
        for _ in range(iterations):
            c_ij = torch.softmax(b_ij, dim=1)
            s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
            v_j = self.squash(s_j)
            if _ < iterations - 1:
                b_ij += (u_hat * v_j).sum(dim=-1, keepdim=True)
        return v_j.squeeze()

    def squash(self,x, dim=-1):
        norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
        norm = norm_sq.sqrt()
        scale = norm_sq / (1 + norm_sq)
        return scale * x / norm

#=====================================================================
#               Modify Models -Regression Task
#=====================================================================
class RREfficientNetRegressor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(RREfficientNetRegressor,self).__init__(*args, **kwargs)
        self.base_model = RREfficientNetClassifier(num_classes=64)
        for layer_param in self.base_model.parameters():
            layer_param.requires_grad = False
        self.base_model.fc1.weight.requires_grad_(True)
        self.base_model.fc2.weight.requires_grad_(True)
        self.base_model.fc1.bias.requires_grad_(True)
        self.base_model.fc2.bias.requires_grad_(True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(64, 1)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='linear')
        self.mean = torch.tensor([0.5, 0.5, 0.5])
        self.std = torch.tensor([0.5, 0.5, 0.5])
        
    def forward(self,x):
        x = F.relu(self.base_model(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x
        
class RRRecognizer(nn.Module):
    def __init__(self):
        super(RRRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 28 * 28, 64)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, 1)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='linear')

        self.mean = torch.tensor([0.5, 0.5, 0.5])
        self.std = torch.tensor([0.5, 0.5, 0.5])

    def forward(self, x):
        # Normalize input
        x = (x - self.mean.to(x.device)[None, :, None, None]) / \
            self.std.to(x.device)[None, :, None, None]

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        # print(x.cpu())
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EfficientNetSpectrom(nn.Module):
    def __init__(self):
        super(EfficientNetSpectrom,self).__init__()
        # self.effnet = EfficientNet.from_pretrained('efficientnet-b0')
        # in_features = self.effnet._fc.out_features
        # self.fc = nn.Linear(in_features,num_classes)
        # self.fc.weight.requires_grad_(True)
        # self.fc.activation = nn.Identity()
        
        self.features = EfficientNet.from_pretrained('efficientnet-b0')
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(1280,512)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512,1)
    
    def forward(self,x):
        # x = self.effnet(x)
        # x = self.fc(x)
        # return torch.round(x)
        
        x = self.features.extract_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        # print(x.cpu())
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        # output = torch.sigmoid(x) * (60 - 8) + 8
        # output = torch.round(output)
        # return output

class LinearRound(nn.Module):
    def __init__(self, in_features, out_features):
        super(EfficientNetSpectrom,self).__init__()
        self.features = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.features._fc.in_features
        
        self.linear = nn.Linear(in_features, out_features)
        # Register the parameters of the linear layer
        self.register_parameter('weight', self.linear.weight)
        self.register_parameter('bias', self.linear.bias)

    def forward(self, x):
        return torch.round(self.linear(x))

class EfficientNetRegressor(nn.Module):
    def __init__(self):
        super(EfficientNetRegressor,self).__init__()
        self.features = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.features._fc.in_features
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features._fc = nn.Linear(in_features,1)
        # self.features._fc.weight.data.normal_(std=0.02)
        # self.features._fc.bias.data.zero_()
        self.features._fc.weight.requires_grad_(True)
        self.features._fc.bias.requires_grad_(True)
        # self.features._fc.activation = nn.Identity()
        # model._fc.activation = nn.Round()

    def forward(self, x):
        output = torch.sigmoid(self.features(x)) * (60 - 8) + 8
        return output
        # output = self.features(x)
        # return output
        # return torch.round(self.features(x))

# model = EfficientNet.from_pretrained('efficientnet-b0')
# in_features = model._fc.in_features
# model._fc = nn.Linear(in_features, 1, bias=True) # Replace last layer with regression task that rounds output to nearest integer
# model._fc = nn.Linear(in_features,num_classes,bias=True) # Replace last layer with target classes
# model._fc.weight.data.normal_(std=0.02)
# model._fc.bias.data.zero_()
# model._fc.weight.requires_grad_(True)
# model._fc.bias.requires_grad_(True)
# model._fc.activation = nn.Identity()
# model._fc.activation = nn.Round()
#=====================================================================
#               End Modify Models
#=====================================================================