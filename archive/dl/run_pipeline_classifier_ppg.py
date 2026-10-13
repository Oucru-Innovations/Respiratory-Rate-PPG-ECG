import torch
# import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import csv
from sklearn.metrics import roc_auc_score
import sys
sys.path.append(".")
# import torchvision.transforms as transforms
from common.utils import one_hot_to_prob
from torch.utils.data import DataLoader
# from efficientnet_pytorch import EfficientNet
from model_pool import STransformerLSTMEffEnsemble,TransformerEncoderLayer_rewrite,\
            RREfficientAttentionNetClassifierPPG,RREfficientNetClassifierPPG
from dl.wearable_data import PPGClassificationDataset
from torch.utils.data import DataLoader
import seaborn as sns
from model_utils import ClosenessLoss, FocalLoss
# import plotly.graph_objs as go
import matplotlib.pyplot as plt
import torch.nn.functional as F
import mlflow
import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
seed = 31
torch.manual_seed(seed)

input_shape = (3, 224, 224) # (channels, height, width)

wavelet_params = {
    'sigma':0.2
}

lr = 1e-3   #6.395137542918557e-05
batch_size = 16
num_epochs = 400000
num_classes = 40
weight_decay = 0.01965630146834162
dropout_rate = 0.3#0.462511285458387
patience = 2000

train_data = [] # List of ECG signals for training

train_fname = 'dataset/ppg_with_resp_from_monitor_wave_ppg_from_monitor/new_ppg_train_data_resp.csv' #public_ppg_data
train_target_fname = 'dataset/ppg_with_resp_from_monitor_wave_ppg_from_monitor/new_ppg_train_target_resp.csv'
val_fname = 'dataset/ppg_with_resp_from_monitor_wave_ppg_from_monitor/new_ppg_val_data_resp.csv'
val_target_fname = 'dataset/ppg_with_resp_from_monitor_wave_ppg_from_monitor/new_ppg_val_target_resp.csv'

# train_fname = 'dataset/public_ppg_train_data_2.csv' #public_ppg_data
# train_target_fname = 'dataset/public_ppg_train_target_2.csv'
# val_fname = 'dataset/public_ppg_val_data_2.csv'
# val_target_fname = 'dataset/public_ppg_val_target_2.csv'

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

train_indices = np.random.permutation(np.arange(len(train_data)))
train_indices = train_indices[:1000]
train_data = np.array(train_data)[train_indices]
train_target = torch.LongTensor((np.array(train_target)[0][train_indices]).reshape(1,-1))
# train_indices = np.random.permutation(len(train_data))
# train_data = np.array(train_data)[train_indices[:len(train_indices)]]
# train_target = torch.LongTensor((np.array(train_target)[0][train_indices[:len(train_indices)]]).reshape(1,-1))
train_set = PPGClassificationDataset(train_data,train_target)

train_data = []
train_target = train_target.cpu().detach().numpy()[0]
train_target = np.eye(num_classes)[train_target]
train_loader = DataLoader(train_set,
                            shuffle=True,
                            batch_size=batch_size, 
                            pin_memory=True, num_workers=8)

train_set = None

val_indices = np.random.permutation(np.arange(len(val_data)))
val_indices = val_indices[:1000]
val_data = np.array(val_data)[val_indices]
val_targets = torch.LongTensor((np.array(val_targets)[0][val_indices]).reshape(1,-1))
# val_indices = np.random.permutation(len(val_data))
# val_data = np.array(val_data)[val_indices[:len(val_indices)]]
# val_targets = torch.LongTensor((np.array(val_targets)[0][val_indices[:len(val_indices)]]).reshape(1,-1))

val_set = PPGClassificationDataset(val_data,val_targets)

val_data = []
val_targets = val_targets.cpu().detach().numpy()[0]
val_targets = np.eye(num_classes)[val_targets]
val_loader = DataLoader(val_set,batch_size=batch_size, 
                            shuffle=True,
                            pin_memory=True, num_workers=8)

val_set = None

# Train the model
# num_train = len(train_set)
# indices = list(range(num_train))
# split = int(0.2 * num_train)
# train_indices, val_indices = indices[split:], indices[:split]

# train_sampler = SubsetRandomSampler(train_indices) 
# val_sampler = SubsetRandomSampler(val_indices)

# train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler) 
# val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler)
# train_loader = DataLoader(train_set,
#                           shuffle=True,
#                           batch_size=batch_size, 
#                           pin_memory=True, num_workers=2)
# val_loader = DataLoader(val_set,batch_size=batch_size, 
#                         shuffle=True,
#                         pin_memory=True, num_workers=2)


# Set Backend CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True # May increase training speed for large models

# Initialize the model and optimizer
# model = RRWaveletClassifier(
#                             input_shape=input_shape,
#                             num_classes=num_classes,
#                             wavelet_params = wavelet_params)
# model = RRClassifier()                                # OK
# model = RRClassifierUNet(num_classes=num_classes)     # OK
# model = RRConvGRUCapsModel(num_classes=num_classes)
# RREfficientAttentionNetClassifierPPG
# RREfficientTransformerNetClassifierPPG
# model = RREfficientAttentionNetClassifierPPG(num_classes=num_classes,fs=100)   #OK
# model = RREfficientNetCapsule(num_classes=num_classes)    # Too heavy
# image_size = 128
# num_channels = 3
image_size = 224
num_channels = 1
lstm_input_dim = 100*30+2
# lstm_input_dim = 180
# num_classes = 4  # Number of classes for your classification task
# model = STransformerLSTMEffEnsemble(image_size, num_classes,
#                                  patch_size=image_size, 
#                                      num_channels=num_channels,
#                                      lstm_input_dim = lstm_input_dim)
# model = RREfficientAttentionNetClassifierPPG(num_classes=num_classes, fs=100)
model = RREfficientNetClassifierPPG(num_classes=num_classes, fs=100)

# Then move your model and data loaders to the device

# Convert model to use float16
# model = model.half()

model.to(device)


# Compute class weights based on the frequency of each class in the dataset
# class_counts = train_set.get_class_counts()  # returns a dictionary of class counts
# total_samples = sum(class_counts.values())
# class_weights = torch.zeros(train_set.num_classes)
# for c in range(train_set.num_classes):
#     if class_counts[c] > 0:
#         class_weights[c] = 1.0 / class_counts[c]
# class_weights = train_set.get_class_weights()

# Define the loss function and optimizer
criterion = ClosenessLoss()#class_weights=class_weights.to(device))
# criterion = FocalLoss()#class_weights=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Use a learning rate scheduler to reduce the learning rate by a factor of 0.1 after 25 and 40 epochs
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=25)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)

# Initialize the plot
sns.set_style('whitegrid')
fig, ax1 = plt.subplots()
# Create a second y-axis that shares the same x-axis with ax1
ax2 = ax1.twinx()

# Train the model
best_val_loss = float('inf') 
patience_counter = 0 

# Initialize lists to store the accuracies
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []
train_auc_list = []
val_auc_list = []

retrain = True
if retrain:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint = torch.load("RREfficientAttentionNetClassifierPPG.pth")
    checkpoint = torch.load("saved_models/ppg_model_latest.pth")
    

    # Access the saved data
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    checkpoint_epoch = checkpoint['epoch']
    loss = checkpoint['best_loss']

    # Create a new model instance and load the saved state dictionary
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
else:
    checkpoint_epoch = -1
    

# optimizer = optim.Adagrad(model.parameters(),lr=lr)

# Start an MLflow run
mlflow.start_run()


# Save the model architecture to a file
model_architecture_path = "/home/khoaldv/mlflow/resp/artifacts/model_architecture.txt"
with open(model_architecture_path, "w") as f:
    print(model, file=f)
# Log the model architecture file as an artifact
# mlflow.log_artifact(model_architecture_path, artifact_path="models")

# Log some parameters and configuration settings
mlflow.log_param("learning_rate", lr)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("weight_decay",weight_decay)
mlflow.log_param("loss",criterion)

accumulation_step = 1

for epoch in tqdm(range(checkpoint_epoch + 1, num_epochs + 1)): 
    # start train
    model.train()
    
    # Initialize running loss and accuracy variables for training set
    train_loss = 0.0
    train_acc = 0.0
    
    counter_i=0
    
    # train_indices = np.random.permutation(len(train_data))
    # # train_target = torch.LongTensor((np.array(train_target)[0][train_indices[:len(train_indices)//10]]).reshape(-1,1))  
    # train_loader = DataLoader(PPGClassificationDataset(np.array(train_data)[train_indices[:len(train_indices)//50]],
    #                             torch.LongTensor((np.array(train_target)[0][train_indices[:len(train_indices)//50]]).reshape(1,-1))),
    #                             shuffle=True,
    #                             batch_size=batch_size, 
    #                             pin_memory=True, num_workers=4)

    train_preds = []
    for original_inp, inputs, targets in train_loader: 
        # Move inputs and targets to the appropriate device
        inputs = inputs.to(device)
        targets = targets.to(device)
        original_inp = original_inp.to(device)
        original_inp = original_inp.unsqueeze(1)
        
        # Zero the gradients to clear accumulated gradients from previous iterations
        optimizer.zero_grad()
        
        # Forward pass through the model to get predicted outputs
        outputs = model((original_inp,inputs))
        # outputs = model(original_inp,inputs)
        
        # loss = criterion(outputs, targets) 
        loss = criterion(outputs,targets)
        # Add regularization to the loss
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += weight_decay * l2_reg
        
        # Backward pass to calculate gradients
        loss.backward()
        
        max_grad_norm = 1.0
        clip_grad_norm_(model.parameters(),max_grad_norm)
        
        if (counter_i+1)%accumulation_step == 0:
            # Update weights using optimizer
            optimizer.step()
            optimizer.zero_grad()
        
        counter_i+=1
        
        train_loss += loss.item() * inputs.size(0) 
        
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.to(device).cpu().numpy().tolist())
        preds_onehot = F.one_hot(preds, num_classes=num_classes)
        train_acc += (targets * preds_onehot).sum()
    
    if (counter_i% accumulation_step)!=0:
        optimizer.step()
        optimizer.zero_grad()
    
    
     # Update the learning rate using the scheduler
    scheduler.step()
    
    # Calculate the average training loss and accuracy for this epoch
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    train_auc = []
    # Calculate ROC AUC for each class
    train_probs = one_hot_to_prob(train_preds)
        
    class_roc_auc_scores = []
    
    for class_idx in range(num_classes):
        # Extract the true labels for the current class
        # class_true_labels = [1 if label == class_idx else 0 for label in train_target]
        
        class_true_labels = train_target[:, class_idx]
        # Extract the predicted probabilities for the current class
        class_predicted_probs =  [prob[class_idx] for prob in train_probs]
            
        # Compute ROC AUC for the current class
        try:
            class_roc_auc = roc_auc_score(class_true_labels, class_predicted_probs)
        except Exception as err:
            class_roc_auc = 0.5
            
        class_roc_auc_scores.append(class_roc_auc)

    # Compute the average ROC AUC across all classes
    train_auc = sum(class_roc_auc_scores) / len(class_roc_auc_scores)


    # Append the metrics to their respective lists
    train_auc_list.append(train_auc)
    
    train_acc_list.append(train_acc.detach().cpu().item())
    train_loss_list.append(train_loss)
    
    # log some metrics for this epoch
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_acc", train_acc, step=epoch)
    mlflow.log_metric("train_auc", train_auc, step=epoch)
    
    
    #======================================================
    # Set the model to evaluation mode
    #======================================================
    model.eval()
    
    
    # val_indices = np.random.permutation(len(val_data))
    # # val_targets = torch.LongTensor((np.array(val_targets)[0][val_indices[:len(val_indices)//10]]).reshape(-1,1))
    # val_loader = DataLoader(PPGClassificationDataset(np.array(val_data)[val_indices[:len(val_indices)//70]],
    #                             torch.LongTensor((np.array(val_targets)[0][val_indices[:len(val_indices)//70]]).reshape(1,-1))),
    #                         batch_size=batch_size, 
    #                         shuffle=True,
    #                         pin_memory=True, num_workers=4)
    
    # Initialize running loss and accuracy variables for validation set
    val_loss = 0.0
    val_acc = 0.0
    
    val_predictions = []

    with torch.no_grad():
        val_preds = []
        for original_inp,inputs, targets in val_loader:
            # Move inputs and targets to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)
            original_inp = original_inp.to(device)
            original_inp = original_inp.unsqueeze(1)
            # Forward pass through the model to get predicted outputs
            outputs = model((original_inp,inputs))
            # outputs = model(original_inp,inputs)
            
            # loss = criterion(outputs, targets)
            loss = criterion(outputs,targets)
            # Add regularization to the loss
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += weight_decay * l2_reg
            
            # Add the loss from this batch to the total validation loss
            val_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            preds_onehot = F.one_hot(preds, num_classes=num_classes)
            val_acc += (targets * preds_onehot).sum()
            val_preds.extend(preds.to(device).cpu().numpy().tolist())
        
        print("=========== Preds - Targets ===========")
        _,targets_idx = torch.max(targets, 1)
        pairs = list(zip(preds.cpu().tolist(), 
                         targets_idx.cpu().tolist()))
        print(pairs)
        print("=============================")
        
        val_auc = []
        # Calculate ROC AUC for each class
        val_probs = one_hot_to_prob(val_preds)
            
        class_roc_auc_scores = []
        
        for class_idx in range(num_classes):
            # Extract the true labels for the current class
            class_true_labels = val_targets[:, class_idx]
                
            # Extract the predicted probabilities for the current class
            class_predicted_probs = [prob[class_idx] for prob in val_probs]
                
            # Compute ROC AUC for the current class
            try:
                class_roc_auc = roc_auc_score(class_true_labels, class_predicted_probs)
            except Exception as err:
                class_roc_auc = 0.5
                
            class_roc_auc_scores.append(class_roc_auc)

        # Compute the average ROC AUC across all classes
        val_auc = sum(class_roc_auc_scores) / len(class_roc_auc_scores)
          
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        
        val_auc_list.append(val_auc)
        val_acc_list.append(val_acc.detach().cpu().item())
        val_loss_list.append(val_loss)

        # log some metrics for this epoch
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("val_auc", val_auc, step=epoch)
        
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Add dropout to the model
    model.dropout_rate = dropout_rate
    
    # Update the plot with the new accuracies
    ax1.clear()
    ax2.clear()
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=train_acc_list, ax=ax1, label='Training accuracy')
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=val_acc_list,ax=ax1, 
                 label='Validation accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Acc')
    # ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 1.5), ncol=2)
    
    train_loss_list = [np.inf if np.isnan(x) else x for x in train_loss_list]
    val_loss_list = [np.inf if np.isnan(x) else x for x in val_loss_list]
    
    # ax2.clear()
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=train_loss_list, 
                 ax=ax2, label='Training loss',
                 color=(0.2,0.2,0.7))
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=val_loss_list,ax=ax2, 
                 label='Validation loss',
                 color=(0.8,0.2,0.2))
    # ax2.set_xlabel('Epoch')
    ax2.yaxis.set_label_position('right')
    ax2.set_ylabel('Loss')
    
    # Add a legend outside the plot
    
    plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.show(block=False)
    plt.pause(0.1)
    
    # Check if validation loss has decreased, otherwise increase patience counter
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'best_loss': best_val_loss}
        
        torch.save(checkpoint, 'RREfficientAttentionNetClassifierPPG.pth')
    else:
        patience_counter += 1

    # Check if early stopping criteria is reached
    if patience_counter >= patience:
        print("Early stopping criterion reached, stopping training.")
        break
    
    
    
    # Save the model after each epoch
    checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'best_loss': best_val_loss}
    
    # torch.save(checkpoint, f"saved_models/ppg_model_clf_{epoch+1}.pth")
    torch.save(checkpoint, f"saved_models/ppg_model_latest.pth")
    
    # save some artifacts for this epoch

    
    plt.savefig("PPG_Plot_Trainning.png")

mlflow.end_run()