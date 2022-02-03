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

test_path = sys.argv[1]
model_path = sys.argv[2]
pred_path = sys.argv[3]

# DataLoader Class
# if BATCH_SIZE = N, dataloader returns images tensor of size [N, C, H, W] and labels [N]
class DevanagariDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform = None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [data, labels] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,:-1].to_numpy()
            labels = data.iloc[:,-1].astype(int)
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
        image = np.array(image).astype(np.uint8).reshape(32, 32, 1)
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
#         print(image.shape, label, type(image))
        sample = {"images": image, "labels": label}
        return sample

# Data Loader Usage

NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

test_data = test_path # Path to test csv file
test_dataset = DevanagariDataset(data_csv = test_data, train=False, img_transform=img_transforms) ######### Make train=False ################
BATCH_SIZE = len(test_dataset.labels) # Batch Size. Adjust accordingly
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS) 

##################################################################################### Check if correct for computer ########################################
if(torch.cuda.is_available()):
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

class convolutional(nn.Module):
  def __init__(self):
    super(convolutional, self).__init__()
    self.CONV1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(3,3),stride=1)
    self.BN1 = nn.BatchNorm2d(32) 
    self.POOL1 = nn.MaxPool2d(2,2)
    self.CONV2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1)
    self.BN2 = nn.BatchNorm2d(64)
    self.POOL2 = nn.MaxPool2d(2,2)
    self.CONV3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=(3,3),stride=1)
    self.BN3 = nn.BatchNorm2d(256)
    self.POOL3 = nn.MaxPool2d(2,1)
    self.CONV4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1)
    self.FC1 = nn.Linear(512 * 1 * 1, 256) 
    self.DROPOUT = nn.Dropout(0.2)
    self.FC2 = nn.Linear(256,46)
    
  def forward(self, inp):
    # 1,32,32
    inp = self.CONV1(inp) # 32,30,30
    inp = self.BN1(inp)
    inp = F.relu(inp)
    inp = self.POOL1(inp) # 32,15,15

    inp = self.CONV2(inp) # 64,13,13
    inp = self.BN2(inp)
    inp = F.relu(inp)
    inp = self.POOL2(inp) # 64,6,6

    inp = self.CONV3(inp) # 256,4,4
    inp = self.BN3(inp)
    inp = F.relu(inp)
    inp = self.POOL3(inp) # 256,3,3

    inp = self.CONV4(inp) # 512,1,1
    inp = F.relu(inp)
    inp = inp.view(-1, 512) 

    inp = self.FC1(inp) # 256
    inp = F.relu(inp)
    
    inp = self.DROPOUT(inp)
    inp = self.FC2(inp) # 46
    
    return inp

convModel = convolutional().to(device) ###################### Check if correct ######################
convModel.load_state_dict(torch.load(model_path))
convModel.eval()

with torch.no_grad():
  for batch_idx, sample in enumerate(test_loader):
    images = sample['images']
    labels = sample['labels']

    images = images.to(device)
    labels = labels.to(device)

    output = convModel(images)
    probs, preds = torch.max(output,1)

    preds = [int(x) for x in preds]

file = open(pred_path, "w")
for number in preds:
    file.write(str(number) + "\n")
file.close()