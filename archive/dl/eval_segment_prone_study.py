import torch 
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from model_pool import RREfficientNetClassifier,RRRecognizer,EfficientNetSpectrom,\
        LinearRound,EfficientNetRegressor,RREfficientNetGMClassifierPPG
from torch.utils.data import DataLoader
from model_utils import loss_uniform_spread_l2
import sys
sys.path.append(".")
from dl.wearable_data import PPGClassificationDataset,\
            ECGDataset,ECGClassificationDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
checkpoint = torch.load("ppg_best_model_clf.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Access the saved data
model_state_dict = checkpoint['model_state_dict']
optimizer_state_dict = checkpoint['optimizer_state_dict']
epoch = checkpoint['epoch']
loss = checkpoint['best_loss']

# Create a new model instance and load the saved state dictionary
# model = RRRecognizer()
# model = RREfficientNetClassifier(num_classes=64,fs=128) 
num_classes = 64
model = RREfficientNetGMClassifierPPG(num_classes=num_classes,fs=100) 
model.load_state_dict(model_state_dict)
model.to(device)
# Create a new optimizer instance and load the saved state dictionary
# optimizer = optimizer(model.parameters(), lr=0.001)
# optimizer.load_state_dict(optimizer_state_dict)

batch_size = 10
num_classes = 64

#============================================
#======= ECG DATA============================
#============================================

# val_data = [] # List of ECG signals for validation
# with open('dataset/val_data.csv', mode='r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         val_data.append([float(x) for x in row])

# val_targets = [] # List of target values for validation
# with open('dataset/val_target.csv', mode='r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         val_targets.append([float(x) for x in row])
        
# # val_set = ECGDataset(val_data,val_targets)
# # val_loader = DataLoader(val_set,batch_size=32, pin_memory=True, num_workers=2)

# val_targets = torch.LongTensor(val_targets)
# val_set = ECGClassificationDataset(val_data,val_targets)

#============================================
#======= PPG DATA============================
#============================================
val_predictions = []
for i in range(7):
    val_data = [] # List of ECG signals for validation
    with open(f'/mnt/Data/Workspace/ProneStudy/notebooks/processed_ppg_data_{str(i)}.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            val_data.append([float(x) for x in row])


    val_set = PPGClassificationDataset(val_data, None, mode='test')
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    val_loss = 0.0
    weight_decay = 0.2
    criterion = nn.MSELoss()


    with torch.no_grad():
        for original_inp,inputs in val_loader:
            # Move inputs and targets to the appropriate device
            inputs = inputs.to(device)
            
            original_inp = original_inp.to(device)
            original_inp = original_inp.unsqueeze(1)
            # Forward pass through the model to get predicted outputs
            outputs = model((original_inp,inputs))
                        
            _, preds = torch.max(outputs, 1)
            preds_onehot = F.one_hot(preds, num_classes=num_classes)
            val_predictions=val_predictions+preds.cpu().tolist()

print(len(val_predictions))
# df = pd.DataFrame()
# df["predict"] = np.apply_along_axis(np.round,axis=0,arr=val_predictions)
# df.to_csv("results/prone_study_res_ppg.csv")
