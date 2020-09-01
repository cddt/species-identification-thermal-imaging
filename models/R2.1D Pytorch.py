import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset

class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]
        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x

class SpatioTemporalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()
        self.downsample = downsample
        padding = kernel_size//2
        if self.downsample:
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))    
        res = self.bn2(self.conv2(res))
        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))
        return self.outrelu(x + res)

class SpatioTemporalResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock, downsample=False):
        super(SpatioTemporalResLayer, self).__init__()
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x

class R2Plus1DNet(nn.Module):
    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DNet, self).__init__()
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)
        self.pool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        return x.view(-1, 512)

class R2Plus1DClassifier(nn.Module):
    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DClassifier, self).__init__()
        self.res2plus1d = R2Plus1DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.res2plus1d(x)
        x = self.linear(x) 
        return x  

def validate(dataset):
    # Returns the number of correct classifications in the validation set
    model.eval()
    correct = 0
    for labels, imgs in dataset:
        output = model(imgs.to(device))
        correct += (output.max(1).indices.cpu() == labels).sum().item()
    return correct

def train(epoch):
    # Performs a single epoch of training
    model.train()
    progress = 0
    for i, (labels, imgs) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(imgs.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()

        if i % 32 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]'.
                format(epoch+1, i * len(labels), len(train_loader.dataset),
                       100. * i / len(train_loader)))

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

print("Dataset loading..", end = " ")
train_imgs = torch.FloatTensor(np.load("./cacophony-preprocessed/training.npy").transpose((0,2,1,3,4)))
train_labels = torch.tensor(np.load("./cacophony-preprocessed/training-labels.npy"))
val_imgs = torch.FloatTensor(np.load("./cacophony-preprocessed/validation.npy").transpose((0,2,1,3,4)))
val_labels = torch.tensor(np.load("./cacophony-preprocessed/validation-labels.npy"))
test_imgs = torch.FloatTensor(np.load("./cacophony-preprocessed/test.npy").transpose((0,2,1,3,4)))
test_labels = torch.tensor(np.load("./cacophony-preprocessed/test-labels.npy"))
print("Dataset loaded!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = R2Plus1DClassifier(num_classes = np.unique(train_labels).size, layer_sizes = [2, 2, 2, 2]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
train_data = TensorDataset(train_labels, train_imgs)
validate_data = TensorDataset(val_labels, val_imgs)
test_data = TensorDataset(test_labels, test_imgs)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(validate_data, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for i in range(epochs):
    start.record()
    #print("Training epoch ", i+1, "..", sep = "")
    train(i)
    print("Training accuracy after", i+1, "epochs:", end = " ")
    print(round(100 * validate(train_loader) / len(train_labels), 1), "%", sep = "")
    print("Validation accuracy after", i+1, "epochs:", end = " ")
    print(round(100 * validate(val_loader) / len(val_labels), 1), "%", sep = "")
    end.record()
    torch.cuda.synchronize()
    print(
        'Epoch {} time: {:.1f} seconds.'.
        format(i+1, start.elapsed_time(end) / 1000)
        )
print("Accuracy on hold out set:", round(100 * test() / len(test_labels), 1), "%", sep = "")
