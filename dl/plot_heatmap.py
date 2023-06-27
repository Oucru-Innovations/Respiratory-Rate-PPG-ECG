import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import csv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import sys
from model_pool import RRClassifierUNet, RREfficientNetClassifier, \
    RRWaveletClassifier, RRClassifier, RRConvGRUCapsModel
from dl.wearable_data import ECGDataset,ECGClassificationDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import seaborn as sns
from model_utils import ClosenessLoss
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import torch.nn.functional as F
import mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.000003
batch_size = 64
num_epochs = 40000
num_classes = 64
weight_decay = 0.05
dropout_rate = 0.2
patience = 100

model = RREfficientNetClassifier(num_classes=num_classes)
model.to(device)

retrain = True
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

checkpoint = torch.load("best_model_clf.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_data = [] # List of ECG signals for validation
with open('dataset/val_data.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        val_data.append([float(x) for x in row])

val_targets = [] # List of target values for validation
with open('dataset/val_target.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        val_targets.append([float(x) for x in row])

val_targets = torch.LongTensor(val_targets)
val_set = ECGClassificationDataset(val_data,val_targets)

val_loader = DataLoader(val_set,batch_size=batch_size, 
                        shuffle=True,
                        pin_memory=True, num_workers=2)

    # Access the saved data
model_state_dict = checkpoint['model_state_dict']
optimizer_state_dict = checkpoint['optimizer_state_dict']
checkpoint_epoch = checkpoint['epoch']
loss = checkpoint['best_loss']

# Create a new model instance and load the saved state dictionary
model.load_state_dict(model_state_dict)
optimizer.load_state_dict(optimizer_state_dict)

model.eval()


with torch.no_grad():
    for inputs, targets in val_loader:
        # Move inputs and targets to the appropriate device
        inputs = inputs.to(device)
        output, feature_map, heatmap = model(inputs,
                                             heatmap_layer=4,
                                             layer_idx=5)
        
        # # Calculate the mean activation value for each channel of the feature map
        # feature_map = layer_output.mean(dim=1, keepdim=True)
        
        # Normalize the mean activations with a softmax function
        # softmax = nn.Softmax(dim=2)
        # heatmap = softmax(feature_map.reshape(feature_map.shape[0], feature_map.shape[1], -1)).reshape(feature_map.shape)

        # Plot the heatmap using Matplotlib
        # plt.imshow(heatmap.squeeze().detach().numpy(), cmap='hot', interpolation='nearest')
        # plt.show()
        # plt.plot(heatmap)
        plt.imshow(np.transpose(feature_map), cmap='jet')
        plt.show()