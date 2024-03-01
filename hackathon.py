# %%
import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Dropout, Linear, ReLU, CrossEntropyLoss
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from torch.nn.parameter import Parameter
import torchvision
import os
from IPython.core.ultratb import AutoFormattedTB
__ITB__ = AutoFormattedTB(mode = 'Verbose',color_scheme='LightBg', tb_offset = 1)

# %%
class BevDataset(Dataset):
  def __init__(self, root, size=512, train=True):
    postfix = 'train' if train else 'test'
    root = os.path.join(root, 'bev_classification', 'images')
    self.dataset_folder = torchvision.datasets.ImageFolder(os.path.join(root, postfix) ,transform = transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor()]))
    # print(self.dataset_folder.class_to_idx)


  def __getitem__(self,index):
    img = self.dataset_folder[index]
    return img[0], img[1]
  
  def __len__(self):
    return len(self.dataset_folder)

# %%
class ImageClassifier(nn.Module):
    def __init__(self, dataset, dropout=0.5):
        super(ImageClassifier, self).__init__()
        output = 99
        self.dropout_ = Dropout(dropout)
        self.dropout = dropout
        self.fc1 = Linear(512*32*32, output*4)
        self.fc2 = Linear(output*4, output)
        x,_ = dataset.dataset[0]
        # print(x.size())
        c,_,_ = x.size()
        self.net = nn.Sequential(
            # Image size = 512 x 512 x 3
            Conv2d(c, 64, (3,3), padding=(1,1)), 
            ReLU(),
            # Image size = 512 x 512 x 64
            Conv2d(64, 64, (3,3), padding=(1,1)), 
            ReLU(),
            Dropout(dropout),
            MaxPool2d(kernel_size=2, stride=2),

            # Image size = 256 x 256 x 64
            Conv2d(64, 128, (3,3), padding=(1,1)), 
            ReLU(),
            # Image size = 256 x 256 x 128
            Conv2d(128, 128, (3,3), padding=(1,1)), 
            ReLU(),
            Dropout(dropout),
            MaxPool2d(kernel_size=2, stride=2),

            # Image size = 128 x 128 x 128
            Conv2d(128, 256, (3,3), padding=(1,1)), 
            ReLU(),
            # Image size = 128 x 128 x 256
            Conv2d(256, 256, (3,3), padding=(1,1)),
            ReLU(),
            Dropout(dropout),
            MaxPool2d(kernel_size=2, stride=2),

            # Image size = 64 x 64 x 256
            Conv2d(256, 512, (3,3), padding=(1,1)),
            ReLU(),
            # Image size = 64 x 64 x 512
            Conv2d(512, 512, (3,3), padding=(1,1)), 
            ReLU(),
            Dropout(dropout),
            MaxPool2d(kernel_size=2, stride=2)
            # Image size = 32 x 32 x 512
        )

    def forward(self, X):
        output = self.net(X)
        _, c, h, w = output.size()
        output = output.view(-1, c*h*w)
        output = self.dropout_(self.fc1(output))
        return self.fc2(output)

# %%
# Hyper parameters
# With batch_size 50, there will be 1776 iterations over the dataset per epoch
batch_size = 1
num_epochs = 100
lr = 1e-3


# %%

def train():
    device = torch.device('mps')
    train_loader = DataLoader(BevDataset('.'), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(BevDataset('.', train=False), batch_size=batch_size, shuffle=True)
    model = ImageClassifier(train_loader).to(device)

    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loop = tqdm(total=len(train_loader)*num_epochs, position=0)

    for epoch in range(num_epochs):
        for x, y_truth in train_loader:
            x, y_truth = x.to(device), y_truth.to(device)

            optimizer.zero_grad()

            y_hat = model(x)
            # print(y_hat.size())
            loss = criterion(y_hat, y_truth)

            loss.backward()
            optimizer.step()
            
            loop.update(1)

train()


# %%



