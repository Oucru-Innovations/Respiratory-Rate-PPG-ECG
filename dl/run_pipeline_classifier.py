import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import csv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from model_pool import RRClassifierUNet, RREfficientNetClassifier, \
    RRWaveletClassifier, RRClassifier, RRConvGRUCapsModel,\
    RREfficientNetCapsule
from dl.wearable_data import ECGDataset,ECGClassificationDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import seaborn as sns
from model_utils import ClosenessLoss
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import torch.nn.functional as F
import mlflow
import sys
import warnings


warnings.filterwarnings("ignore")

sys.path.append(".")
# Set random seed for reproducibility
seed = 17
torch.manual_seed(seed)


input_shape = (3, 224, 224) # (channels, height, width)

wavelet_params = {
    'sigma':0.2
}



lr = 0.000005
batch_size = 64
num_epochs = 40000
num_classes = 64
weight_decay = 0.01
dropout_rate = 0.3
patience = 100


train_data = [] # List of ECG signals for training
with open('dataset/train_data.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        train_data.append([float(x) for x in row])

train_target = [] # List of target values for training
with open('dataset/train_target.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        train_target.append([float(x) for x in row])

train_target = torch.LongTensor(train_target)

train_set = ECGClassificationDataset(train_data,train_target)

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

train_loader = DataLoader(train_set,
                          shuffle=True,
                          batch_size=batch_size, 
                          pin_memory=True, num_workers=2)
val_loader = DataLoader(val_set,batch_size=batch_size, 
                        shuffle=True,
                        pin_memory=True, num_workers=2)


# Train the model
# num_train = len(train_set)
# indices = list(range(num_train))
# split = int(0.2 * num_train)
# train_indices, val_indices = indices[split:], indices[:split]

# train_sampler = SubsetRandomSampler(train_indices) 
# val_sampler = SubsetRandomSampler(val_indices)

# train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler) 
# val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler)
train_loader = DataLoader(train_set,
                          shuffle=True,
                          batch_size=batch_size, 
                          pin_memory=True, num_workers=2)
val_loader = DataLoader(val_set,batch_size=batch_size, 
                        shuffle=True,
                        pin_memory=True, num_workers=2)


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
model = RREfficientNetClassifier(num_classes=num_classes,fs=128)   #OK
# model = RREfficientNetCapsule(num_classes=num_classes)    # Too heavy

# Then move your model and data loaders to the device
model.to(device)

# Define the loss function and optimizer
criterion = ClosenessLoss() 
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

retrain = True
if retrain:
    checkpoint = torch.load("best_model_clf.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model_architecture_path = "artifacts/model_architecture.txt"
with open(model_architecture_path, "w") as f:
    print(model, file=f)
# Log the model architecture file as an artifact
mlflow.log_artifact(model_architecture_path, artifact_path="models")

# Log some parameters and configuration settings
mlflow.log_param("learning_rate", lr)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("weight_decay",weight_decay)
mlflow.log_param("loss",criterion)

for epoch in tqdm(range(checkpoint_epoch + 1, num_epochs + 1)): 
    # start train
    model.train()
    
    # Initialize running loss and accuracy variables for training set
    train_loss = 0.0
    train_acc = 0.0
    
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
        
        # loss = criterion(outputs, targets) 
        loss = criterion(outputs,targets)
        # Add regularization to the loss
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += weight_decay * l2_reg
        
        # Backward pass to calculate gradients
        loss.backward()

        # Update weights using optimizer
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0) 
        
        _, preds = torch.max(outputs, 1)
        preds_onehot = F.one_hot(preds, num_classes=num_classes)
        train_acc += (targets * preds_onehot).sum()
    

     # Update the learning rate using the scheduler
    scheduler.step()
    
    # Calculate the average training loss and accuracy for this epoch
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    train_acc_list.append(train_acc.detach().cpu().item())
    train_loss_list.append(train_loss)
    
    # log some metrics for this epoch
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_acc", train_acc, step=epoch)
    
    
    #======================================================
    # Set the model to evaluation mode
    #======================================================
    model.eval()
    
    # Initialize running loss and accuracy variables for validation set
    val_loss = 0.0
    val_acc = 0.0
    
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for original_inp,inputs, targets in val_loader:
            # Move inputs and targets to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)
            original_inp = original_inp.to(device)
            original_inp = original_inp.unsqueeze(1)
            # Forward pass through the model to get predicted outputs
            outputs = model((original_inp,inputs))
            
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
        
        print("=========== Preds - Targets ===========")
        _,targets_idx = torch.max(targets, 1)
        pairs = list(zip(preds.cpu().tolist(), 
                         targets_idx.cpu().tolist()))
        print(pairs)
        print("=============================")
          
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        val_acc_list.append(val_acc.detach().cpu().item())
        val_loss_list.append(val_loss)

        # log some metrics for this epoch
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Add dropout to the model
    model.dropout_rate = dropout_rate
    
    # Update the plot with the new accuracies
    ax1.clear()
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=train_acc_list, ax=ax1, label='Training accuracy')
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=val_acc_list,ax=ax1, label='Validation accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Acc')
    
    ax2.clear()
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=train_loss_list, 
                 ax=ax2, label='Training loss',
                 color=(0.2,0.2,0.7))
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=val_loss_list,ax=ax2, 
                 label='Validation loss',
                 color=(0.8,0.2,0.2))
    # ax2.set_xlabel('Epoch')
    ax2.yaxis.set_label_position('right')
    ax2.set_ylabel('Loss')
    
    plt.legend()
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
        
        torch.save(checkpoint, 'best_model_clf_ecg.pth')
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
    
    torch.save(checkpoint, f"saved_models/model_clf_{epoch+1}_ecg.pth")
    
    # save some artifacts for this epoch

    
    plt.savefig("Plot_Trainning_ECG.png")

mlflow.end_run()