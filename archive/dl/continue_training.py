import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from dl.wearable_data import ECGDataset
from sklearn.metrics import mean_squared_error,accuracy_score,mean_absolute_error
from model_pool import RREfficientNetRegressor,RRRecognizer,EfficientNetSpectrom,LinearRound,EfficientNetRegressor
from model_utils import loss_uniform_spread_l2
import csv
import warnings
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import matplotlib.pyplot as plt
import mlflow
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Define hyperparameters
lr = 0.000003#0.0000005
batch_size = 128+64
num_epochs = 2000
num_classes = 1
weight_decay = 0.01
dropout_rate = 0.2
patience = 6

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

train_target = torch.FloatTensor(train_target)

train_set = ECGDataset(train_data,train_target)

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


val_targets = torch.FloatTensor(val_targets)

# Create validation dataset
val_set = ECGDataset(val_data,val_targets)

# train_loader = DataLoader(train_set,batch_size=8)
# val_loader = DataLoader(val_set,batch_size=8)
train_loader = DataLoader(train_set,
                          shuffle=True,
                          batch_size=batch_size, 
                          pin_memory=True, num_workers=2)
val_loader = DataLoader(val_set,batch_size=batch_size, 
                        shuffle=True,
                        pin_memory=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True # May increase training speed for large models

#==================================================
#               Define Model
#==================================================
model = RREfficientNetRegressor()  # OK

checkpoint = torch.load("best_model_clf.pth")
# Access the saved data
model_state_dict = checkpoint['model_state_dict']
optimizer_state_dict = checkpoint['optimizer_state_dict']
epoch = checkpoint['epoch']
loss = checkpoint['best_loss']
checkpoint_epoch = -1

# model = EfficientNetSpectrom() # OK

# Start an MLflow run
mlflow.start_run()
model_architecture_path = "artifacts/model_architecture.txt"
with open(model_architecture_path, "w") as f:
    print(model, file=f)
# Log the model architecture file as an artifact
mlflow.log_artifact(model_architecture_path, artifact_path="models")

mlflow.log_artifact(model_architecture_path, artifact_path="models")

# Log some parameters and configuration settings
mlflow.log_param("learning_rate", lr)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("weight_decay",weight_decay)

# Then move your model and data loaders to the device
model.to(device)
model.base_model.load_state_dict(model_state_dict)


# Use Adam optimizer with weight decay and a learning rate of 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, 
                             weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, 
#                              weight_decay=weight_decay,momentum=0.9)
# optimizer.load_state_dict(optimizer_state_dict)
# Define loss function (MSE)
criterion = loss_uniform_spread_l2()
# criterion = nn.L1Loss()
# criterion = nn.CrossEntropyLoss()

# Use a learning rate scheduler to reduce the learning rate by a factor of 0.1 after 25 and 40 epochs
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=25)

# Define early stopping
best_val_loss = float('inf')
patience_counter = 0

# Initialize lists to store the accuracies
train_mae_list = []
val_mae_list = []
train_loss_list = []
val_loss_list = []

# Initialize the plot
sns.set_style('whitegrid')
fig, ax1 = plt.subplots()
# Create a second y-axis that shares the same x-axis with ax1
ax2 = ax1.twinx()


for epoch in tqdm(range(num_epochs)):
    # start train
    model.train()
    
    # Initialize running loss and accuracy variables for training set
    train_loss = 0.0
    train_acc = 0.0
    
    for inputs, targets in train_loader:
        # Move inputs and targets to the appropriate device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the gradients to clear accumulated gradients from previous iterations
        optimizer.zero_grad()
        
        # Forward pass through the model to get predicted outputs
        outputs = model(inputs)                   #model(inputs.cuda())
        # outputs = model(x.unsqueeze(0)) 
        
        # Calculate the loss by comparing predicted outputs with target labels
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
        
        train_loss += loss.item()*inputs.size(0)
        
        # Calculate the accuracy for this batch and add it to the total training accuracy
        preds,_ = torch.max(outputs, dim=1)
        # train_acc += accuracy_score(targets.cpu(), preds.cpu()) * inputs.size(0)
        train_acc += mean_absolute_error(targets.cpu(), preds.cpu().detach().numpy()) * inputs.size(0)
    
    # print(preds.cpu())
    
    # Update the learning rate using the scheduler
    scheduler.step()
    
    # Calculate the average training loss and accuracy for this epoch
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    train_mae_list.append(train_acc)
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
        for inputs, targets in val_loader:
            # Move inputs and targets to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass through the model to get predicted outputs
            outputs = model(inputs)
            # outputs = model(inputs.cuda())
            
            # Calculate the loss by comparing predicted outputs with target labels
            # loss = criterion(outputs, targets)
            loss = criterion(outputs,targets)
            # loss = criterion(outputs, targets.cuda())
            # Add regularization to the loss
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += weight_decay * l2_reg
            
            # Add the loss from this batch to the total validation loss
            val_loss += loss.item() * inputs.size(0)

            preds,_ = torch.max(outputs, dim=1)
            # val_acc += accuracy_score(targets.cpu(), preds.cpu()) * inputs.size(0)
            val_acc += mean_absolute_error(targets.cpu(), preds.cpu()) * inputs.size(0)
        print("=========== Preds - Targets ===========")
        predict_rounded = [round(x,1) for x in preds.cpu().tolist()]

        pairs = list(zip(predict_rounded, 
                         targets.cpu().tolist()))
        print(pairs)
        print("=============================")
            
            # val_predictions.append(outputs.cpu().numpy())
            # val_targets.append(targets.numpy())
        # val_loss /= len(val_loader.dataset)
        # val_predictions = np.concatenate(val_predictions)
        # val_targets = np.concatenate(val_targets)
        # val_mse = mean_squared_error(val_predictions,val_targets)
        # val_mse = 0
        
    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    
    # log some metrics for this epoch
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_acc", val_acc, step=epoch)
        
    
    val_mae_list.append(val_acc)
    val_loss_list.append(val_loss)
    
    # Add dropout to the model
    model.dropout_rate = dropout_rate
    
    # Update the plot with the new accuracies
    ax1.clear()
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=train_mae_list, ax=ax1, label='Training accuracy')
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=val_mae_list,ax=ax1, label='Validation accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Acc')
    
    ax2.clear()
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=train_loss_list, 
                 ax=ax2, label='Training loss',
                 color=(0.8,0.2,0.2))
    sns.lineplot(x=range(checkpoint_epoch + 1,epoch + 1), y=val_loss_list,ax=ax2, 
                 label='Validation loss',
                 color=(0.2,0.2,0.7))
    # ax2.set_xlabel('Epoch')
    ax2.yaxis.set_label_position('right')
    ax2.set_ylabel('Loss')
    
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)
    
    # Check if validation loss has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        #reset patience counter
        patience_counter = 0
        # save the model checkpoint
        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'best_loss': best_val_loss}
        
        torch.save(checkpoint, 'best_model.pth')

    else:
        patience_counter += 1

    # Early stopping condition
    if patience_counter >= patience:
        print(f"Validation loss has not improved for {patience} epochs. Stopping early.")
        break

    
    # Print the training and validation loss and accuracy for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train MAE: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_acc:.4f}")

    # Save the model after each epoch
    torch.save(model.state_dict(), f"saved_models/model_{epoch+1}_reg.pt")