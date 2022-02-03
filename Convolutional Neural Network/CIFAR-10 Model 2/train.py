import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import glob
import os
import collections
import torchvision
import pandas as pd
from skimage import io, transform
import scipy
import sklearn
import matplotlib.pyplot as plt # for plotting
import cv2

from tqdm import tqdm
from IPython.display import Image

import sys

train_path = sys.argv[1]
test_path = sys.argv[2]
model_path = sys.argv[3]
loss_path = sys.argv[4]
accuracy_path = sys.argv[5]

# DataLoader Class
# if BATCH_SIZE = N, dataloader returns images tensor of size [N, C, H, W] and labels [N]
class ImageDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform=None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [labels,data] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,1:].to_numpy()
            labels = data.iloc[:,0].astype(int)
        else:
            images = data.iloc[:,:].to_numpy()
            labels = None
        
        self.images = images
        self.labels = labels
        print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Loads image of the given index and performs preprocessing.
        
        INPUT: 
        idx: index of the image to be loaded.
        
        OUTPUT:
        sample: dictionary with keys images (Tensor of shape [1,C,H,W]) and labels (Tensor of labels [1]).
        """
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape((32, 32, 3),order="F")
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
        
        sample = {"images": image, "labels": label}
        return sample

# Data Loader Usage

BATCH_SIZE = 200 # Batch Size. Adjust accordingly
NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

# Train DataLoader
train_data = train_path # Path to train csv file
train_dataset = ImageDataset(data_csv = train_data, train=True, img_transform=img_transforms)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

# Test DataLoader
test_data = test_path # Path to test csv file
test_dataset = ImageDataset(data_csv = test_data, train=True, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

##################################################################################### Check if correct for computer ########################################
if(torch.cuda.is_available()):
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

#Checkpoint1
class convolutional(nn.Module):
  def __init__(self):
    super(convolutional, self).__init__()
    self.CONV1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),stride=1)
    self.BN1 = nn.BatchNorm2d(32) 
    self.POOL1 = nn.MaxPool2d(2,2)
    self.CONV2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=(4,4),stride=1)
    self.BN2 = nn.BatchNorm2d(128)
    self.POOL2 = nn.MaxPool2d(2,2)
    self.CONV3 = nn.Conv2d(in_channels=128,out_channels=512,kernel_size=(3,3),stride=1)
    self.BN3 = nn.BatchNorm2d(512)
    self.POOL3 = nn.MaxPool2d(2,2)
    self.CONV4 = nn.Conv2d(in_channels=512,out_channels=1000,kernel_size=(2,2),stride=1)
    self.FC1 = nn.Linear(1000 * 1 * 1, 256) 
    self.DROPOUT = nn.Dropout(0.5)
    self.FC2 = nn.Linear(256,10)
    
  def forward(self, inp):
    # 1,32,32
    inp = self.CONV1(inp) # 32,30,30
    inp = self.BN1(inp)
    inp = F.relu(inp)
    inp = self.POOL1(inp) # 32,15,15

    inp = self.CONV2(inp) # 128,12,12
    inp = self.BN2(inp)
    inp = F.relu(inp)
    inp = self.POOL2(inp) # 128,6,6

    inp = self.CONV3(inp) # 512,4,4
    inp = self.BN3(inp)
    inp = F.relu(inp)
    inp = self.POOL3(inp) # 512,2,2

    inp = self.CONV4(inp) # 1000,1,1
    inp = F.relu(inp)
    inp = inp.view(-1, 1000) 

    inp = self.FC1(inp) # 256
    inp = F.relu(inp)
    
    inp = self.DROPOUT(inp)
    inp = self.FC2(inp) # 10
    
    return inp

torch.manual_seed(51)
convModel = convolutional().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=convModel.parameters(),lr=1e-3)

epochs = 20

accuracies = []
losses = []
last_epoch = 0
for epoch in range(last_epoch+1,epochs+1):
  count = 0
  sum_loss = 0
  for batch_idx, sample in enumerate(train_loader):
    count += 1
    images = sample['images']
    labels = sample['labels']

    images = images.to(device)
    labels = labels.to(device)

    output = convModel(images)
    loss = loss_function(output,labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    sum_loss += loss.item()
  
  losses += [sum_loss/count]

  count = 0
  sum_accuracy = 0
  convModel.eval()
  with torch.no_grad():
    for batch_idx, sample in enumerate(test_loader):
      count += 1
      images = sample['images']
      labels = sample['labels']

      images = images.to(device)
      labels = labels.to(device)

      output = convModel(images)
      probs, preds = torch.max(output,1)

      sum_accuracy += (preds==labels).sum().item()/BATCH_SIZE
  convModel.train()
  accuracies += [sum_accuracy/count] 
  print(accuracies[-1])
  if(epoch>=10 and accuracies[-1]>0.76):
    last_epoch = epoch
    break

optimizer = torch.optim.Adam(params=convModel.parameters(),lr=1e-4)

for epoch in range(last_epoch+1,epochs+1):
  count = 0
  sum_loss = 0
  for batch_idx, sample in enumerate(train_loader):
    count += 1
    images = sample['images']
    labels = sample['labels']

    images = images.to(device)
    labels = labels.to(device)

    output = convModel(images)
    loss = loss_function(output,labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    sum_loss += loss.item()
  
  losses += [sum_loss/count]

  count = 0
  sum_accuracy = 0
  convModel.eval()
  with torch.no_grad():
    for batch_idx, sample in enumerate(test_loader):
      count += 1
      images = sample['images']
      labels = sample['labels']

      images = images.to(device)
      labels = labels.to(device)

      output = convModel(images)
      probs, preds = torch.max(output,1)

      sum_accuracy += (preds==labels).sum().item()/BATCH_SIZE
  convModel.train()
  accuracies += [sum_accuracy/count] 
  print(accuracies[-1])

torch.save(convModel.state_dict(), model_path)

file = open(loss_path, "w")
for number in losses:
    file.write(str(number) + "\n")
file.close()

file = open(accuracy_path, "w")
for number in accuracies:
    file.write(str(number) + "\n")
file.close()