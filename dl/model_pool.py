import torch 
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np
from layer_pool import DigitCapsules, CapsuleLayer, CapsuleNetwork, WaveletLayer
import biosppy.signals.ecg as ecg
from scipy import signal
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from vit_pytorch import ViT  # Import ViT from lucidrains/vit-pytorch
from vit_pytorch.extractor import Extractor
import timm
import torchvision.models as models


#=====================================================================
#               Modify Models - Classification Task
#=====================================================================
import torch
import torch.nn as nn
from vit_pytorch import ViT

class LiteViTRegressor(nn.Module):
    def __init__(self, fs=100, image_size=224, patch_size=8, channels=3, 
                dim=64, depth=2, heads=4, mlp_dim=128, dropout=0.1, emb_dropout=0.1):
        super(LiteViTRegressor, self).__init__()
        self.fs = fs
        
        # Lightweight ViT
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=64,  # Output dim-sized features
            dim=dim,         # Smaller embedding size
            depth=depth,     # Fewer layers
            heads=heads,     # Fewer attention heads
            mlp_dim=mlp_dim, # Smaller MLP
            channels=channels,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        self.vit_out_dim = dim  # Matches dim
        
        # Simplified regression head
        self.regressor_head = nn.Sequential(
            nn.Linear(self.vit_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)  # Raw RR output
        )

    def forward(self, x, return_attention=False):
        _, spectrogram = x  # [batch, 3, 224, 224]
        if return_attention:
            features, attn_weights = self.vit(spectrogram, return_attn=True)  # [batch, 64]
            output = self.regressor_head(features)
            return output, attn_weights
        else:
            features = self.vit(spectrogram)  # [batch, 64]
            output = self.regressor_head(features)
            return output

class ViTRegressor(nn.Module):
    def __init__(self,
                 pretrained=False,
                 pretrained_model_name='vit_base_patch16_224',
                 fs=100,
                 image_size=224,
                 patch_size=8,  # Smaller patches for finer detail
                 channels=3,
                 dim=256,       # More capacity
                 depth=6,       # Deeper for better feature extraction
                 heads=8,       # More attention heads
                 mlp_dim=512,   # Larger MLP for richer processing
                 dropout=0.1,
                 emb_dropout=0.1):
        super(ViTRegressor, self).__init__()
        self.fs = fs
        self.pretrained = pretrained

        if pretrained:
            import timm
            self.vit = timm.create_model(pretrained_model_name, pretrained=True, num_classes=0)  # No head
            self.vit_out_dim = 256
        else:
            self.vit = ViT(
                image_size=image_size,
                patch_size=patch_size,
                num_classes=256,  # No direct output; use features
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                channels=channels,
                dropout=dropout,
                emb_dropout=emb_dropout
            )
            self.vit_out_dim = 256

        # Enhanced regression head
        self.regressor_head = nn.Sequential(
            nn.Linear(self.vit_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Raw output
        )

    def forward(self, x, return_attention=False):
        _, spectrogram = x
        if self.pretrained:
            features = self.vit(spectrogram)  # [batch, embed_dim]
            output = self.regressor_head(features)
            return output
        else:
            if return_attention:
                features, attn_weights = self.vit(spectrogram, return_attn=True)
                output = self.regressor_head(features)
                return output, attn_weights
            else:
                features = self.vit(spectrogram)
                output = self.regressor_head(features)
                return output

class CNNLSTMRegressor(nn.Module):
    def __init__(self, fs=100, input_channels=3, hidden_dim=128, num_layers=2):
        super(CNNLSTMRegressor, self).__init__()
        self.fs = fs
        
        # CNN for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [32, 112, 112]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [64, 56, 56]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # [128, 28, 28]
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(input_size=128 * 28, hidden_size=hidden_dim, 
                        num_layers=num_layers, batch_first=True)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        _, spectrogram = x  # [batch, 3, 224, 224]
        batch_size = spectrogram.size(0)
        
        # CNN feature extraction
        features = self.cnn(spectrogram)  # [batch, 128, 28, 28]
        features = features.view(batch_size, 28, -1)  # [batch, 28, 128*28]
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)  # [batch, 28, hidden_dim]
        lstm_out = lstm_out[:, -1, :]  # Take last time step [batch, hidden_dim]
        
        # Regression
        output = self.regressor(lstm_out)  # [batch, 1]
        return output

class LiteTransformerRegressor(nn.Module):
    def __init__(self, fs=100, image_size=224, patch_size=16, channels=3, 
                dim=64, depth=2, heads=4, mlp_dim=128, dropout=0.1):
        super(LiteTransformerRegressor, self).__init__()
        self.fs = fs
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 196 patches
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)  # [batch, dim, 14, 14]
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim))
        
        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)
        
        # Global pooling and regression
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)  # Raw RR output
        )

    def forward(self, x):
        _, spectrogram = x  # [batch, 3, 224, 224]
        batch_size = spectrogram.size(0)
        
        # Patch embedding
        patches = self.patch_embed(spectrogram)  # [batch, dim, 14, 14]
        patches = patches.flatten(2).transpose(1, 2)  # [batch, 196, dim]
        patches = patches + self.pos_embed  # Add positional encoding
        
        # Transformer processing
        transformer_out = self.transformer(patches)  # [batch, 196, dim]
        
        # Pool and regress
        pooled = self.pool(transformer_out.transpose(1, 2)).squeeze(-1)  # [batch, dim]
        output = self.regressor(pooled)  # [batch, 1]
        return output

class WaveNetRegressor(nn.Module):
    def __init__(self, fs=100, input_channels=3):
        super(WaveNetRegressor, self).__init__()
        self.fs = fs
        
        # Dilated conv layers for long-range dependencies
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(input_channels if i == 0 else 64, 64, kernel_size=3, padding=2**i, dilation=2**i)
            for i in range(3)  # 3 layers with dilations 1, 2, 4
        ])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(3)])
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(3)])  # Downsample
        
        # Global pooling and regression
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)  # Raw RR output
        )

    def forward(self, x):
        _, spectrogram = x  # [batch, 3, 224, 224]
        features = spectrogram
        for conv, act, pool in zip(self.conv_layers, self.activations, self.pools):
            features = conv(features)
            features = act(features)
            features = pool(features)  # [batch, 32, smaller dims]
        
        features = self.pool(features)  # [batch, 32, 1, 1]
        features = features.view(features.size(0), -1)  # [batch, 32]
        output = self.regressor(features)  # [batch, 1]
        return output

class LiteCNNRegressor(nn.Module):
    def __init__(self, fs=100, input_channels=3):
        super(LiteCNNRegressor, self).__init__()
        self.fs = fs
        
        # CNN with large kernels to capture low frequencies
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=7, padding=3),  # Large kernel
            nn.ReLU(),
            nn.MaxPool2d(2),  # [16, 112, 112]
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [32, 56, 56]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [64, 28, 28]
            nn.AdaptiveAvgPool2d(1)  # [64, 1, 1]
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)  # Raw RR output
        )

    def forward(self, x):
        _, spectrogram = x  # [batch, 3, 224, 224]
        features = self.cnn(spectrogram)  # [batch, 64, 1, 1]
        features = features.view(features.size(0), -1)  # [batch, 64]
        output = self.regressor(features)  # [batch, 1]
        return output

class ConvNeXtRegressor(nn.Module):
    def __init__(self, fs=100, pretrained=True):
        super(ConvNeXtRegressor, self).__init__()
        self.fs = fs
        
        # Use ConvNeXt-Tiny with default weights
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        # Remove classifier to get features
        self.convnext.classifier = nn.Identity()
        
        # Reduce feature dim and regress
        self.regressor = nn.Sequential(
            nn.Linear(768, 128),  # ConvNeXt-Tiny outputs 768 features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Raw RR output
        )

    def forward(self, x):
        _, spectrogram = x  # [batch, 3, 224, 224]
        features = self.convnext(spectrogram)  # [batch, 768, 1, 1]
        features = features.view(features.size(0), -1)  # [batch, 768], flatten spatial dims
        output = self.regressor(features)  # [batch, 1]
        return output
    
class WaveViTRegressor(nn.Module):
    def __init__(self, fs=100, input_channels=3, dim=64, heads=4, depth=1, dropout=0.1):
        super(WaveViTRegressor, self).__init__()
        self.fs = fs
        
        # WaveNet-inspired dilated convs
        self.wavenet = nn.ModuleList([
            nn.Conv2d(input_channels if i == 0 else 32, 32, kernel_size=3, padding=2**i, dilation=2**i)
            for i in range(3)  # Dilations 1, 2, 4
        ])
        self.wavenet_acts = nn.ModuleList([nn.ReLU() for _ in range(3)])
        self.wavenet_pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(3)])  # [32, 28, 28] after
        
        # Lightweight ViT-like transformer
        self.num_patches = 28 * 28  # After pooling
        self.patch_to_embedding = nn.Conv2d(32, dim, kernel_size=1)  # [batch, dim, 28, 28]
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim))
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*2, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)
        
        # Regression head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)  # Raw RR output
        )

    def forward(self, x):
        _, spectrogram = x  # [batch, 3, 224, 224]
        batch_size = spectrogram.size(0)
        
        # WaveNet feature extraction
        features = spectrogram
        for conv, act, pool in zip(self.wavenet, self.wavenet_acts, self.wavenet_pools):
            features = conv(features)
            features = act(features)
            features = pool(features)  # [batch, 32, 28, 28]
        
        # ViT-like transformer processing
        patches = self.patch_to_embedding(features)  # [batch, dim, 28, 28]
        patches = patches.flatten(2).transpose(1, 2)  # [batch, 784, dim]
        patches = patches + self.pos_embed  # Add positional encoding
        transformer_out = self.transformer(patches)  # [batch, 784, dim]
        
        # Pool and regress
        pooled = self.pool(transformer_out.transpose(1, 2)).squeeze(-1)  # [batch, dim]
        output = self.regressor(pooled)  # [batch, 1]
        return output
    
class ViTWaveRegressor(nn.Module):
    def __init__(self, fs=100, image_size=224, patch_size=16, channels=3, 
                 dim=64, depth=2, heads=4, mlp_dim=128, dropout=0.1, emb_dropout=0.1):
        super(ViTWaveRegressor, self).__init__()
        self.fs = fs
        
        # Lightweight ViT first
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=dim,  # No direct output, use embeddings
            dim=dim,        # Small embedding size
            depth=depth,    # Few layers
            heads=heads,    # Few attention heads
            mlp_dim=mlp_dim,# Small MLP
            channels=channels,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 196 for 16x16
        
        # Reshape ViT output back to spatial form
        self.to_spatial = nn.Linear(dim, 32 * (image_size // patch_size) * (image_size // patch_size))
        
        # WaveNet-inspired dilated convs
        self.wavenet = nn.ModuleList([
            nn.Conv2d(32 if i == 0 else 32, 32, kernel_size=3, padding=2**i, dilation=2**i)
            for i in range(3)  # Dilations 1, 2, 4
        ])
        self.wavenet_acts = nn.ModuleList([nn.ReLU() for _ in range(3)])
        self.wavenet_pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(3)])  # [32, 28, 28] after
        
        # Regression head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)  # Raw RR output
        )

    def forward(self, x):
        _, spectrogram = x  # [batch, 3, 224, 224]
        batch_size = spectrogram.size(0)
        
        # ViT feature extraction
        vit_out = self.vit(spectrogram)  # [batch, dim]
        spatial_features = self.to_spatial(vit_out)  # [batch, 32*14*14]
        spatial_features = spatial_features.view(batch_size, 32, 14, 14)  # [batch, 32, 14, 14]
        
        # WaveNet processing
        features = spatial_features
        for conv, act, pool in zip(self.wavenet, self.wavenet_acts, self.wavenet_pools):
            features = conv(features)
            features = act(features)
            features = pool(features)  # [batch, 32, size shrinks]
        
        # Pool and regress
        pooled = self.pool(features).view(batch_size, -1)  # [batch, 32]
        output = self.regressor(pooled)  # [batch, 1]
        return output

