# %%
from torch.nn import Conv2d, MaxPool2d, Dropout, Linear, ReLU, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torchvision
import pathlib
import pickle
import torch
import os
import gc

# %%
class BevDataset(Dataset):
  def __init__(self, root, size=512, split='train', chunk=0):
    postfix = f'{split}/{split}_{chunk}'
    root = os.path.join(root, 'bev_classification', 'images')
    self.dataset_folder = torchvision.datasets.ImageFolder(os.path.join(root, postfix) ,transform = transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor()]))

  def __getitem__(self,index):
    img = self.dataset_folder[index]
    return img[0], img[1]
  
  def __len__(self):
    return len(self.dataset_folder)

# %%
class ImageClassifier(nn.Module):
    def __init__(self, dropout=0.2):
        super(ImageClassifier, self).__init__()
        output = 99
        self.dropout = Dropout(dropout)
        self.fc1 = Linear(512*32*32, output)
        self.fc2 = Linear(output, output)
        self.conv1 = Conv2d(3, 64, (3,3), padding=(1,1))
        self.conv2 = Conv2d(64, 128, (3,3), padding=(1,1))
        self.conv3 = Conv2d(128, 256, (3,3), padding=(1,1))
        self.conv4 = Conv2d(256, 512, (3,3), padding=(1,1))
        
        self.net = nn.Sequential(
            # Image size = 512 x 512 x 3
            self.conv1, 
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            # Image size = 256 x 256 x 64
            self.conv2, 
            ReLU(),
            Dropout(dropout),
            MaxPool2d(kernel_size=2, stride=2),

            # Image size = 128 x 128 x 128
            self.conv3, 
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            # Image size = 64 x 64 x 256
            self.conv4,
            ReLU(),
            Dropout(dropout),
            MaxPool2d(kernel_size=2, stride=2)
            # Image size = 32 x 32 x 512
        )

        self.initialize_weights()


    def initialize_weights(self):
        gain = 2**(1/2)
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)
        nn.init.xavier_normal_(self.conv1.weight, gain=gain)
        nn.init.xavier_normal_(self.conv2.weight, gain=gain)
        nn.init.xavier_normal_(self.conv3.weight, gain=gain)
        nn.init.xavier_normal_(self.conv4.weight, gain=gain)


    def forward(self, X):
        output = self.net(X)
        _, c, h, w = output.size()
        output = output.view(-1, c*h*w)
        output = self.dropout(self.fc1(output))
        return self.fc2(output)

# %%
# Hyper parameters
# With batch_size 50, there will be 1776 iterations over the dataset per epoch
num_chunks = 10
batch_size = 4
num_epochs = 3
lr = 1e-2

# %%
train_losses = [0]
train_accuracy = [0]

val_losses = [0]
val_accuracy = [0]

model_path = 'model/mps-model.pkl'

if not pathlib.Path('model').exists():
    pathlib.Path('model').mkdir()

def train():
    try:
        gc.collect()
        device = torch.device('mps')
        if not pathlib.Path(model_path).exists():
            model = ImageClassifier().to(device)
        else:
            print('Model Found!')
            print('Loading model...')
            with open(model_path, 'rb') as f:
                model = pickle.load(f).to(device)

        # TODO: make sure to split the data into 10 samples and train on each 
        for i in range(num_chunks):
            print()
            train_loader = DataLoader(BevDataset('.', chunk=i), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(BevDataset('.', split='val', chunk=i), batch_size=batch_size, shuffle=True)
            
            # Only have 5 validation checks per epoch
            val_check = len(train_loader) // 5

            criterion = CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            loop = tqdm(total=len(train_loader)*num_epochs, position=0)

            for epoch in range(num_epochs):
                train_step_losses = []
                train_step_accuracy = []
                for batch, (x, y_truth) in enumerate(train_loader):
                    x, y_truth = x.to(device), y_truth.to(device)

                    optimizer.zero_grad()

                    y_hat = model(x)
                    accuracy = (y_hat.argmax(1) == y_truth).float().mean()
                    train_step_accuracy.append(accuracy.item())

                    loss = criterion(y_hat, y_truth)
                    train_step_losses.append(loss.item())

                    loss.backward()
                    optimizer.step()

                    if (batch + 1) % val_check == 0:
                        print('Validation check')
                        val_loop = tqdm(total=len(val_loader), position=0)

                        val_batch_loss = []
                        val_batch_accuracy = []
                        for batch, (x, y_truth) in enumerate(val_loader):
                            x, y_truth = x.to(device), y_truth.to(device)

                            optimizer.zero_grad()

                            y_hat = model(x)

                            accuracy = (y_hat.argmax(1) == y_truth).float().mean()
                            val_batch_accuracy.append(accuracy.item())

                            loss = criterion(y_hat, y_truth)
                            val_batch_loss.append(loss.item())

                            loss.backward()
                            optimizer.step()
                            val_loop.update(1)
                            val_loop.set_description(f'val batch: {batch} val accuracy: {accuracy*100:.2f}% val loss: {loss:.4f}')

                        val_losses.append(sum(val_batch_loss) / len(val_batch_loss))
                        val_accuracy.append(sum(val_batch_accuracy) / len(val_batch_accuracy))

                        train_losses.append(sum(train_step_losses) / len(train_step_losses))
                        train_accuracy.append(sum(train_step_accuracy) / len(train_step_accuracy))

                    loop.update(1)
                    loop.set_description(f'epoch: {epoch+1} batch: {batch} accuracy: {train_accuracy[-1]*100:.2f}% val accuracy {val_accuracy[-1]*100:.2f}% loss: {train_losses[-1]:.4f} val loss: {val_losses[-1]:.4f}')
                
                print('Saving model...')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print('Model saved.')
    except:
        print('Saving model...')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print('Model saved.')


train()


# %%



