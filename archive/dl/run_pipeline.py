import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from dl.wearable_data import ECGDataset
from sklearn.metrics import mean_squared_error,accuracy_score,mean_absolute_error
from model_pool import RRRecognizer,EfficientNetSpectrom,LinearRound,EfficientNetRegressor
from model_utils import loss_uniform_spread_l2
import csv
import warnings
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Define hyperparameters
lr = 0.00005#0.0000005
batch_size = 70
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
# train_target = list(map(str,
#                        list(map(int,train_target[0]))
#                 ))
# train_target = torch.LongTensor(train_target)
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

# val_targets = list(map(str,
#                        list(map(int,val_targets[0]))
#                 ))

# Convert target values to a tensor of integers
# val_targets = torch.LongTensor(val_targets)
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
model = RRRecognizer()  # OK

checkpoint = torch.load("best_model.pth")
# Access the saved data
model_state_dict = checkpoint['model_state_dict']
optimizer_state_dict = checkpoint['optimizer_state_dict']
epoch = checkpoint['epoch']
loss = checkpoint['best_loss']

model.load_state_dict(model_state_dict)

# model = EfficientNetSpectrom() # OK
# model = EfficientNet.from_pretrained('efficientnet-b0')
# in_features = model._fc.in_features
# model._fc = nn.Linear(in_features, 1, bias=True) # Replace last layer with regression task that rounds output to nearest integer
# model._fc.weight.data.normal_(std=0.02)
# model._fc.bias.data.zero_()
# model._fc.weight.requires_grad_(True)
# model._fc.bias.requires_grad_(True)
# model._fc.activation = nn.Identity()


# Then move your model and data loaders to the device
model.to(device)

# Use Adam optimizer with weight decay and a learning rate of 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, 
                             weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, 
#                              weight_decay=weight_decay,momentum=0.9)
optimizer.load_state_dict(optimizer_state_dict)
# Define loss function (MSE)
criterion = nn.MSELoss()
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

# Initialize the plot
sns.set_style('whitegrid')
fig, ax = plt.subplots()

for epoch in range(num_epochs):
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
        loss = loss_uniform_spread_l2(outputs,targets)
        
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
            loss = loss_uniform_spread_l2(outputs,targets)
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
    
    val_mae_list.append(val_acc)
    
    # Add dropout to the model
    model.dropout_rate = dropout_rate
    
    # Update the plot with the new accuracies
    ax.clear()
    sns.lineplot(x=range(epoch+1), y=train_mae_list, label='Training accuracy', ax=ax)
    sns.lineplot(x=range(epoch+1), y=val_mae_list, label='Validation accuracy', ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
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
    torch.save(model.state_dict(), f"saved_models/model_{epoch+1}.pt")