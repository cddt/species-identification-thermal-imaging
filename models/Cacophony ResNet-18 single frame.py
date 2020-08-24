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

def validate(dataset):
    # Returns the number of correct classifications in the validation set
    model.eval()
    correct = 0
    for labels, imgs in dataset:
        output = model(imgs.to(device))
        correct += (output.max(1).indices.cpu() == labels).sum().item()
    return correct

def train():
    # Performs a single epoch of training
    model.train()
    progress = 0
    for i, (labels, imgs) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(imgs.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()
        progress += batch_size
        #if (progress / batch_size) % 1000 == 0:
        #    print("\t", round(100 * progress / len(train_labels), 1), "%", sep = "")

def test(): 
    model.eval()
    correct = 0
    for labels, imgs in test_loader:
        output = model(imgs.to(device))
        correct += (output.max(1).indices.cpu() == labels).sum().item()
    return correct
        
epochs = 10
batch_size = 32
learning_rate = 0.001

# Since this baseline is essentially just image classification, we flatten the dataset's time dimension
print("Dataset loading..", end = " ")
train_imgs = torch.FloatTensor(np.load("./cacophony-preprocessed/training.npy")[:,22].reshape([-1, 3, 24, 24]))
train_labels = torch.tensor(np.load("./cacophony-preprocessed/training-labels.npy"))
val_imgs = torch.FloatTensor(np.load("./cacophony-preprocessed/validation.npy")[:,22].reshape([-1, 3, 24, 24]))
val_labels = torch.tensor(np.load("./cacophony-preprocessed/validation-labels.npy"))
test_imgs = torch.FloatTensor(np.load("./cacophony-preprocessed/test.npy")[:,22].reshape([-1, 3, 24, 24]))
test_labels = torch.tensor(np.load("./cacophony-preprocessed/test-labels.npy"))
print("Dataset loaded!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = resnet18(np.unique(train_labels).size, in_channels = 3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
train_data = TensorDataset(train_labels, train_imgs)
validate_data = TensorDataset(val_labels, val_imgs)
test_data = TensorDataset(test_labels, test_imgs)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(validate_data, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

for i in range(epochs):
    print("Training epoch ", i+1, "..", sep = "")
    train()
    print("Training accuracy after", i+1, "epochs:", end = " ")
    print(round(100 * validate(train_loader) / len(train_labels), 1), "%", sep = "")
    print("Validation accuracy after", i+1, "epochs:", end = " ")
    print(round(100 * validate(val_loader) / len(val_labels), 1), "%", sep = "")

print("Accuracy on hold out set:", round(100 * test() / len(test_labels), 1), "%", sep = "")
