import torch
# import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import csv
import os
from sklearn.metrics import roc_auc_score
import sys
sys.path.append(".")
# import torchvision.transforms as transforms
from common.utils import one_hot_to_prob
from torch.utils.data import DataLoader
# from efficientnet_pytorch import EfficientNet
from model_pool import STransformerLSTMEffEnsemble,RREfficientAttentionNetClassifierPPG
from dl.wearable_data import PPGClassificationDataset
from torch.utils.data import DataLoader
import seaborn as sns
from model_utils import ClosenessLoss, FocalLoss
# import plotly.graph_objs as go
import matplotlib.pyplot as plt
import torch.nn.functional as F
import mlflow
import numpy as np
from torch.nn.utils import clip_grad_norm_
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
seed = 19
torch.manual_seed(seed)


input_shape = (3, 224, 224) # (channels, height, width)

wavelet_params = {
    'sigma':0.2
}



lr = 0.0000001
batch_size = 16
num_epochs = 40000
num_classes = 40
weight_decay = 0.005
dropout_rate = 0.25
patience = 2000


train_data = [] # List of ECG signals for training

train_fname = 'dataset/ppg_train_data.csv' #public_ppg_data
train_target_fname = 'dataset/ppg_train_target.csv'
val_fname = 'dataset/ppg_val_data.csv'
val_target_fname = 'dataset/ppg_val_target.csv'

# train_fname = 'dataset/public_ppg_data.csv' #public_ppg_data
# train_target_fname = 'dataset/public_ppg_target.csv'
# val_fname = 'dataset/public_ppg_data.csv'
# val_target_fname = 'dataset/public_ppg_target.csv'

with open(train_fname, mode='r') as file: #ppg_train_data
    reader = csv.reader(file)
    for row in reader:
        train_data.append([float(x) for x in row])

train_target = [] # List of target values for training
with open(train_target_fname, mode='r') as file: #ppg_train_target
    reader = csv.reader(file)
    for row in reader:
        train_target.append([float(x) for x in row])


val_data = [] # List of ECG signals for validation
with open(val_fname, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        val_data.append([float(x) for x in row])

val_targets = [] # List of target values for validation
with open(val_target_fname, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        val_targets.append([float(x) for x in row])

train_indices = np.random.permutation(len(train_data))
train_data = np.array(train_data)[train_indices[:len(train_indices)]]
train_target = (np.array(train_target)[0][train_indices[:len(train_indices)]]).reshape(1,-1)

val_indices = np.random.permutation(len(val_data))
val_data = np.array(val_data)[val_indices[:len(val_indices)]]
val_targets = (np.array(val_targets)[0][val_indices[:len(val_indices)]]).reshape(1,-1)


dta_len = 10000

# np.array(train_data)[:dta_len].tofile("dataset/public_ppg_train_data.csv")
# train_target[0][:dta_len].tofile("dataset/public_ppg_train_target.csv")
# np.array(val_data)[:dta_len].tofile("dataset/public_ppg_val_data.csv")
# val_targets[0][:dta_len].tofile("dataset/public_ppg_val_target.csv")

train_data = (train_data[:dta_len]).tolist()
with open(os.path.join('dataset','public_ppg_train_data_3.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(train_data)
train_target = (train_target.reshape(-1)[:dta_len]).astype(int).tolist()
with open(os.path.join('dataset','public_ppg_train_target_3.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(train_target)
            
val_data = (val_data[:dta_len]).tolist()
with open(os.path.join('dataset','public_ppg_val_data_3.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(train_data)
val_targets = (val_targets.reshape(-1)[:dta_len]).astype(int).tolist()
with open(os.path.join('dataset','public_ppg_val_target_3.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(val_targets)