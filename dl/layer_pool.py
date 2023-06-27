import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import numpy as np
import pywt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.fft

class WaveletLayer(nn.Module):
    def __init__(self, num_scales=3, scale_factor=2, wavelet='morlet', freq=0.25):
        super(WaveletLayer, self).__init__()
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        self.wavelet = wavelet
        self.freq = freq

        # Define the wavelet filter
        if wavelet == 'morlet':
            self.filter_type = 'complex'
            self.filter_size = 7
            self.sigma = 2 * np.pi / freq / (1 + np.sqrt(2))
            self.kernels = self._make_morlet_filter(self.sigma, self.filter_size)
        else:
            raise NotImplementedError('Unsupported wavelet type')

    def forward(self, x):
        # Apply the wavelet transform to the input tensor
        coeffs_list = []
        for i in range(self.num_scales):
            # Compute the size and sigma of the wavelet filter at this scale
            filter_size = self.filter_size * self.scale_factor ** i
            sigma = self.sigma * self.scale_factor ** i

            # Construct the wavelet filter kernel and its Fourier transform
            kernel = self._make_wavelet_kernel(filter_size, sigma)
            kernel_fft = torch.fft.rfft2(kernel, s=x.shape[-2:]).unsqueeze(1)

            # Convolve the input tensor with the wavelet filter kernel
            expanded_kernel = kernel_fft.expand(x.shape[0], x.shape[1], -1, -1)
            expanded_x = x.unsqueeze(1).expand_as(expanded_kernel)
            conv = torch.fft.irfft2(torch.fft.rfft2(expanded_x) * expanded_kernel, s=(x[-2], x[-1]))

            h, l = conv[:, :x.shape[1]//2], conv[:, x.shape[1]//2:]

            # Store the detail and approximation coefficients
            coeffs_list.extend([h, l])

            # Downsample the approximation coefficients for the next scale
            x = nn.functional.avg_pool2d(l, kernel_size=2, stride=2)

        # Concatenate the outputs of the wavelet transform and return
        return torch.cat(coeffs_list + [x], dim=1)
    
    def _make_wavelet_kernel(self, size, sigma):
        if self.filter_type == 'complex':
            kernel_real = self._make_morlet_filter(sigma, size)
            kernel_imag = self._make_morlet_filter(sigma, size, imaginary=True)
            kernel = torch.stack([kernel_real, kernel_imag], dim=0)
        else:
            raise NotImplementedError('Unsupported filter type')
        return kernel
    
    @staticmethod
    def _make_morlet_filter(sigma, size, imaginary=False):
        t = np.linspace(-size // 2, size // 2, size)
        y, x = np.meshgrid(t, t, indexing='ij')
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)

        if imaginary:
            re = np.zeros((size, size))
        else:
            re = np.exp(-(r ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * r / sigma - np.pi) * np.sqrt(np.pi) / sigma

        im = np.exp(-(r ** 2) / (2 * sigma ** 2)) * np.sin(2 * np.pi * r / sigma - np.pi) * np.sqrt(np.pi) / sigma
        kernel = re + 1j * im

        return torch.tensor(kernel, dtype=torch.float32)

class WaveletLayerDL(nn.Module):
    def __init__(self, in_channels, num_orientations=4, kernel_size=3, stride=1, padding=1, wavelet_type='morlet', wavelet_params=None):
        super(WaveletLayer, self).__init__()

        # Define the wavelet filters
        if wavelet_type == 'morlet':
            sigma = wavelet_params['sigma']
            self.psi_filter = self._get_morlet_filter(num_orientations, kernel_size, sigma)
        elif wavelet_type == 'haar':
            self.psi_filter = self._get_haar_filter(num_orientations, kernel_size)
        elif wavelet_type == 'custom':
            filter_fn = wavelet_params.get('filter_fn', None)
            if filter_fn is None:
                raise ValueError('Must provide a valid filter function for custom wavelets')
            self.psi_filter = filter_fn(num_orientations, kernel_size)
        else:
            raise ValueError(f'Unsupported wavelet type: {wavelet_type}')

        # Create the convolutional layers for psi and phi filters
        self.conv_psi = nn.Conv2d(in_channels, num_orientations * 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_phi = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # Apply the wavelet transform to the input tensor
        psi_out = self.conv_psi(x)
        phi_out = self.conv_phi(x)

        # Apply the psi and phi filters to the output of the convolutional layers
        psi_out = self._apply_psi_filter(psi_out, self.psi_filter)
        phi_out = self._apply_phi_filter(phi_out)

        # Concatenate the outputs and return the result
        out = torch.cat([psi_out, phi_out], dim=1)

        return out

    def _get_morlet_filter(self, num_orientations, kernel_size, sigma=2.0):
        # Compute the complex Morlet wavelets
        f0 = 2 / (1 + np.sqrt(2))
        w_real = np.zeros((num_orientations, kernel_size, kernel_size))
        w_imag = np.zeros((num_orientations, kernel_size, kernel_size))
        for i in range(num_orientations):
            theta = i * np.pi / num_orientations
            x, y = np.meshgrid(np.arange(-kernel_size//2 + 1, kernel_size//2 + 1), np.arange(-kernel_size//2 + 1, kernel_size//2 + 1))
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)
            w_real[i] = np.exp(-0.5 * (x_theta ** 2 + y_theta ** 2) / sigma ** 2) * np.cos(2 * np.pi * f0 * x_theta)
            w_imag[i] = np.exp(-0.5 * (x_theta ** 2 + y_theta ** 2) / sigma ** 2) * np.sin(2 * np.pi * f0 * x_theta)

        # Stack the real and imaginary parts and convert to torch tensor
        psi_filter = np.stack([w_real, w_imag], axis=1)
        psi_filter = torch.from_numpy(psi_filter).float()

        return psi_filter

    def _get_haar_filter(self, num_orientations, kernel_size):
        # Compute the filter coefficients for the Haar wavelet
        w_real = np.array([1, -1])
        w_imag = np.zeros_like(w_real)

        # Stack the real and imaginary parts and convert to torch tensor
        psi_filter = np.stack([w_real, w_imag], axis=0).repeat(num_orientations, axis=0).reshape(num_orientations, 2, 1, kernel_size)
        psi_filter = torch.from_numpy(psi_filter).float()

        return psi_filter

    
    def _apply_psi_filter(self, x, psi_filter):
        # Apply the wavelet filters to the input tensor
        num_orientations = psi_filter.shape[0]
        num_channels = x.shape[1]
        psi_out = []
        for i in range(num_orientations):
            psi_weight = psi_filter[i].unsqueeze(0).repeat(num_channels, 1, 1, 1)
            psi_real = nn.functional.conv2d(x[:, :num_channels//2], psi_weight[:, :1], stride=1, padding=(psi_filter.size(-1)//2))
            psi_imag = nn.functional.conv2d(x[:, num_channels//2:], psi_weight[:, 1:], stride=1, padding=(psi_filter.size(-1)//2))
            psi_out.append(psi_real - psi_imag)
        # Concatenate the output of each orientation and return the result
        psi_out = torch.cat(psi_out, dim=1)

        return psi_out
    
    def _apply_phi_filter(self, x):
        # Compute the average of the input tensor along the channel dimension
        phi_out = x.mean(dim=1, keepdim=True)
        return phi_out

class WaveletTransform(nn.Module):
    def __init__(self, wavelet_name='haar', mode='symmetric', level=1):
        super(WaveletTransform, self).__init__()
        self.wavelet_name = wavelet_name
        self.mode = mode
        self.level = level
        
        # Define the wavelet filter bank
        self.filters = torch.Tensor(pywt.Wavelet(self.wavelet_name).filter_bank[0])

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        
        # Reshape the input tensor into a matrix
        matrix = x.view(batch_size * num_channels, 1, height, width)
        
        # Perform horizontal and vertical convolutions using the wavelet filter bank
        low_freqs = F.conv2d(matrix, self.filters[:, None], padding=(0, 1), groups=num_channels)
        high_freqs = F.conv2d(matrix, self.filters[:, None], padding=(0, 1), groups=num_channels)[:, :, :, 1:]
        
        # Extract the approximation and detail coefficients
        approx = F.avg_pool2d(low_freqs, kernel_size=2)
        detail = high_freqs.transpose(2, 3)
        
        # Reshape the coefficients into a tensor and concatenate them along the channel dimension
        coeffs = torch.cat([approx.view(batch_size, num_channels, -1), detail.view(batch_size, num_channels, -1)], dim=-1)
        
        return coeffs
    

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(PrimaryCapsules, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1, 8) # reshape to (batch_size, num_capsules, capsule_dim)
        x = self.squash(x)
        return x
    
    def squash(self, x):
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm)

class DigitCapsules(nn.Module):
    def __init__(self, in_capsules, in_capsule_dim, num_classes, out_capsules, out_capsule_dim):
        super(DigitCapsules, self).__init__()

        self.in_capsules = in_capsules
        self.in_capsule_dim = in_capsule_dim
        self.num_classes = num_classes
        self.out_capsules = out_capsules
        self.out_capsule_dim = out_capsule_dim

        # Define weight matrix for each capsule in the output layer
        self.W = nn.Parameter(torch.randn(num_classes, in_capsules, out_capsules, out_capsule_dim, in_capsule_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Compute predicted capsules by multiplying input capsules with weight matrix and summing over input capsules
        u_hat = torch.matmul(self.W, x.unsqueeze(2)).squeeze(-1)

        # Apply dynamic routing algorithm to compute output capsules
        b = torch.zeros(x.size(0), self.num_classes, self.in_capsules, 1)
        if x.is_cuda:
            b = b.cuda()
        for i in range(3):  # We keep the number of routing iterations fixed to 3
            # Apply softmax to coefficients over classes
            c = self.softmax(b)

            # Compute weighted sum of predicted capsules for each class
            s = (c * u_hat).sum(dim=2, keepdim=True)

            # Apply squash function to get output capsules
            v = self.squash(s)

            # Update logits for next iteration of routing
            if i != 2:
                b = b + torch.matmul(u_hat, v.transpose(1, 2))

        return v.squeeze(-1)

    def squash(self, x):
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm)

class CapsuleLayer(nn.Module):
    def __init__(self, in_capsules, in_capsule_dim, num_classes, out_capsules, out_capsule_dim, in_channels):
        super(CapsuleLayer, self).__init__()
        
        self.in_capsules = in_capsules
        self.in_capsule_dim = in_capsule_dim
        self.num_classes = num_classes
        self.out_capsules = out_capsules
        self.out_capsule_dim = out_capsule_dim
        
        # Define weight matrix for each capsule in the output layer
        self.W = nn.Parameter(torch.randn(num_classes, in_channels, in_capsules, out_capsules, out_capsule_dim, in_capsule_dim))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Reshape input features for compatibility with weight matrix
        x = x.unsqueeze(1).unsqueeze(-1)

        # Compute predicted capsules by multiplying input capsules with weight matrix and summing over input capsules
        u_hat = torch.matmul(self.W, x).squeeze().sum(dim=-1)
        
        # Apply dynamic routing algorithm to compute output capsules
        b = torch.zeros(x.size(0), self.num_classes, self.in_capsules, 1)
        for i in range(3): # number of routing iterations
            c = self.softmax(b)
            s = (c * u_hat).sum(dim=2)
            v = self.squash(s)
            if i < 2:
                b += (u_hat * v.unsqueeze(2)).sum(dim=-1, keepdim=True)
            
        return v
    
    def squash(self, x):
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm)


# Define the Capsule Network model
class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(CapsuleNetwork, self).__init__()

        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        
        # Primary capsules layer
        self.primary_capsules = nn.Conv2d(in_channels=256, out_channels=32*8, kernel_size=9, stride=2)
        
        # Digit capsules layer
        self.digit_capsules = nn.Sequential(
            nn.Linear(32*8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes*16)
        )

    def forward(self, x):
        # Apply convolutional layer with ReLU activation function
        x = F.relu(self.conv1(x))
        
        # Apply primary capsules layer with squash non-linearity
        x = self.primary_capsules(x)
        x = x.view(x.size(0), -1, 8)
        x = self.squash(x)
        
        # Apply digit capsules layer with dynamic routing
        x = self.digit_capsules(x)
        x = x.view(-1, self.num_classes, 16)
        x = self.routing(x)
        logits = (x ** 2).sum(dim=-1) ** 0.5
        
        return logits
    
    # Squash non-linearity
    def squash(self, tensor):
        norm_sq = (tensor ** 2).sum(dim=-1, keepdim=True)
        norm = norm_sq.sqrt()
        return (norm_sq / (1.0 + norm_sq)) * (tensor / norm)

    # Dynamic Routing
    def routing(self, x):
        b = torch.zeros_like(x)
        for i in range(3):
            c = F.softmax(b, dim=1)
            s = (c.unsqueeze(-1) * x).sum(dim=1)
            v = self.squash(s)
            b += (x * v.unsqueeze(1)).sum(-1, keepdim=True)
        return v

