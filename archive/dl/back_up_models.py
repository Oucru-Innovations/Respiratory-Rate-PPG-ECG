import torch 
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np
import pywt

class RREfficientNetClassifier(nn.Module):
    def __init__(self,num_classes, *args, **kwargs) -> None:
        super(RREfficientNetClassifier, self).__init__(*args, **kwargs)
        
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # Freeze the weights of the base model
        for param in self.base_model.parameters():
            param.requires_grad = True
        
        self.num_feats = self.base_model._fc.in_features * 2
        self.fc1 = nn.Linear(self.num_feats, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)        
    
    def forward(self, x, heatmap_layer=None, layer_idx=None):
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
        
        x = self.base_model._avg_pooling(features)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if (layer_idx is not None) or (heatmap_layer is not None):
           
            return x, feature_map, heatmap
        return x
    
class RREfficientNetClassifier(nn.Module):
    def __init__(self,num_classes, *args, **kwargs) -> None:
        super(RREfficientNetClassifier, self).__init__(*args, **kwargs)
        
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # Freeze the weights of the base model
        for param in self.base_model.parameters():
            param.requires_grad = True
        
        self.base_model_ca = EfficientNet.from_pretrained('efficientnet-b0')
        # Freeze the weights of the base model
        for param in self.base_model_ca.parameters():
            param.requires_grad = True
        
        self.num_feats = self.base_model._fc.in_features * 2
        self.fc1 = nn.Linear(self.num_feats, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    # def compute_CtA(self,x):
    #     # convert the PyTorch tensor to a numpy array
    #     ecg_data_np = x.cpu().numpy()
    #     nperseg = 256
    #     noverlap = 128
    #     nfft = 2048
    #     hop_length = nperseg-noverlap
    #     # loop over each batch in the input tensor
    #     for batch in ecg_data_np:
        
    #         signal = torch.istft(batch, n_fft=nfft, hop_length=hop_length)
    #         rr = get_rr(signal,fs=224)
    #     return rr
    
    def wavelet_scan(self,x):
        # assume ecg_data is a PyTorch tensor containing the ECG signal
        # sample_rate is the sampling rate of the ECG signal
        wavelet_name = 'db4'  # select a wavelet type
        level = 2  # select the level of decomposition

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
        
                
        batch_size, levels, channels, w,wavelength = coeffs_tensor.shape
        
        level_1_coeffs = coeffs_tensor[:,0,:,:,:].clone()
        level_1_coeffs = level_1_coeffs.view(batch_size,channels,w,wavelength)
        return level_1_coeffs

        ca_coeffs_tensor = coeffs_tensor[:,:,0,:,:].clone()
        ca_coeffs_tensor = ca_coeffs_tensor.view(batch_size,levels,w, wavelength)
        
        cD_coeffs_tensor = coeffs_tensor[:,:,1,:,:].clone()
        cD_coeffs_tensor = cD_coeffs_tensor.view(batch_size,levels,w, wavelength)
        
        
        # Shape is [batch_size, level of decomposition,number of coefficient arrays at each level of decomposition, height dimension of each spectrogram image,idth dimension of each spectrogram image]

        # Decomposes the signal into multiple frequency bands (approximations and details) at different resolution levels.

        # This dimension represents the number of coefficient arrays at each level of decomposition. Specifically, it corresponds to the approximation coefficients (cA) and the two detail coefficients (cD) for each level of decomposition. For example, at level 1 there are 3 coefficient arrays: cA1, cD1 (horizontal), and cD1 (vertical). The actual number of coefficient arrays may vary depending on the wavelet family used and the level of decomposition.
        
        # length of the wavelet coefficients at each level of decomposition..

        return ca_coeffs_tensor,cD_coeffs_tensor # shape is [batch_size,]
        
    
    def forward(self, x, heatmap_layer=None, layer_idx=None):
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
        ca_features = self.wavelet_scan(x)
        ca_features = self.base_model_ca.extract_features(ca_features.cuda())
        # combine features
        # x = self.base_model._avg_pooling(features)
        # x = x.flatten(start_dim=1)
        
        x_features = self.base_model._avg_pooling(features)
        x_ca_features = self.base_model_ca._avg_pooling(ca_features)
        x_features = x_features.flatten(start_dim=1)
        x_ca_features = x_ca_features.flatten(start_dim=1)
        x = torch.cat([x_features, x_ca_features], dim=1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if (layer_idx is not None) or (heatmap_layer is not None):
           
            return x, feature_map, heatmap
        return x
    

class RREfficientNetClassifier(nn.Module):
    def __init__(self,num_classes,nperseg=256,noverlap=128,nfft=2048,
                 fs=224,
                 *args, **kwargs) -> None:
        super(RREfficientNetClassifier, self).__init__(*args, **kwargs)
        
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.fs = fs
        # Freeze the weights of the base model
        for param in self.base_model.parameters():
            param.requires_grad = True
        
        self.num_feats = self.base_model._fc.in_features * 2
        self.fc1 = nn.Linear(self.num_feats, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)        
    
    def compute_RR(self,x):
        # convert the PyTorch tensor to a numpy array
        ecg_data_np = x.cpu().numpy()
        hop_length = self.nperseg-self.noverlap
        # loop over each batch in the input tensor
        for batch in ecg_data_np:
            # convert back to signal from tensor
            ecg_signal = torch.istft(batch, n_fft=self.nfft, hop_length=hop_length)
            
            # filter signal
            ecg_signal = signal.detrend(ecg_signal) # remove baseline
            b,a = signal.butter(2,[0.1, 0.5], btype="bandpass",fs=self.fs)
            ecg_signal = signal.filtfilt(b,a,ecg_signal)
            
            #compute RR
            rpeaks,_ = ecg.hamilton_segmenter(ecg_signal,self.fs)
            rr_intervals = np.diff(rpeaks)/self.fs
            
            #Calculate PSD
            f, psd = signal.welch(rr_intervals, fs=self.fs, nperseg= 1024)
            
            #Identify RR frequency component
            resp_freq_range = (0.1,0.8)
            resp_freq_mask = np.logical_and(f >= resp_freq_range[0], f <= resp_freq_range[1])
            max_resp_freq_idx = np.argmax(psd[resp_freq_mask]) 
            resp_rate = f[resp_freq_mask][max_resp_freq_idx] * 60
            
        return resp_rate
    
    def forward(self, x, heatmap_layer=None, layer_idx=None):
        
        rr = self.compute_RR(x)
        device = x.device
        rr.to(device)
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
        
        x = self.base_model._avg_pooling(features)
        x = x.flatten(start_dim=1)
        
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if (layer_idx is not None) or (heatmap_layer is not None):
           
            return x, feature_map, heatmap
        return x
