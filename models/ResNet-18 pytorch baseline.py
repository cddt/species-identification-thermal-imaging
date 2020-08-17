import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsat.models.classification import resnet18
from torch.utils.data import DataLoader, TensorDataset

"""
Trains a simple baseline which predicts the class from only a single frame.
Uses the ResNet-18 architecture with random weight initialization (no pre-training).
"""

def validate():
    # Returns the number of correct classifications in the validation set
    model.eval()
    correct = 0
    for labels, imgs in val_loader:
        output = model(imgs.to(device))
        correct += (output.max(1).indices.cpu() == labels).sum().item()
    return correct

def train():
    # Performs a single epoch of training
    model.train()
    progress = 0
    for labels, imgs in train_loader:
        optimizer.zero_grad()
        output = model(imgs.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()
        progress += batch_size
        if (progress / batch_size) % 500 == 0:
            print("\t", round(100 * progress / len(train_labels), 1), "%", sep = "")
        
epochs = 10
batch_size = 32
learning_rate = 0.001

# Since this baseline is essentially just image classification, we flatten the dataset's time dimension
print("Dataset loading..", end = " ")
train_imgs = torch.FloatTensor(np.load("./cacophony-preprocessed/training.npy").reshape([-1, 3, 24, 24]))
train_labels = torch.tensor(np.load("./cacophony-preprocessed/training-labels.npy").repeat(45))
val_imgs = torch.FloatTensor(np.load("./cacophony-preprocessed/validation.npy").reshape([-1, 3, 24, 24]))
val_labels = torch.tensor(np.load("./cacophony-preprocessed/validation-labels.npy").repeat(45))
print("Dataset loaded!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = resnet18(17, in_channels = 3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
train_data = TensorDataset(train_labels, train_imgs)
validate_data = TensorDataset(val_labels, val_imgs)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(validate_data, batch_size = batch_size, shuffle = False)

for i in range(epochs):
    print("Training epoch ", i+1, "..", sep = "")
    train()
    print("Accuracy after", i+1, "epochs:", end = " ")
    print(round(100 * validate() / len(val_labels), 1), "%", sep = "")
