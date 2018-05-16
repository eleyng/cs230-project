import random
import os
import torchvision.transforms as transforms
import flow_transforms
import torch
import torch.utils.data as data
import os.path
from scipy.ndimage import imread
import numpy as np
from torchvision import utils
import matplotlib
matplotlib.use('Agg')
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from torch.autograd import Variable
import PIL
from PIL import Image
import torch.nn as nn
from scipy import misc
import matplotlib.cm as cm
import torch.optim as optim
from model import *

class ListDataset(data.Dataset):
    def __init__(self,data_dir,listing,input_transform=None,target_depth_transform=None,
                target_labels_transform=None,co_transform=None,random_scale = None):

        self.data_dir = data_dir
        self.listing = listing
        #self.depth_imgs = depth_imgs
        self.input_transform = input_transform
        self.target_depth_transform = target_depth_transform
        #self.target_labels_transform = target_labels_transform
        self.co_transform = co_transform

    def __getitem__(self, index):
        img_name = self.listing[index]

        input_dir,target_depth_dir = self.data_dir

        input_im, target_depth_im = imread(os.path.join(input_dir,img_name)),\
                                                    imread(os.path.join(target_depth_dir,img_name[:-4]+'_instanceIds.png'))


        if self.co_transform is not None:
            input_im, target_depth_im = self.co_transform(input_im,target_depth_im)

        if self.input_transform is not None:
            input_im = self.input_transform(input_im)

        if self.target_depth_transform is not None :
            target_depth_im = self.target_depth_transform(target_depth_im)

        # if self.target_labels_transform is not None :
        #     target_label_im = self.target_labels_transform(target_label_im)

        input_rgb_im = input_im
        #input_depth_im  = torch.cat((target_depth_im,target_depth_im,target_depth_im),dim = 0)
        target_im = target_depth_im

        return input_rgb_im,target_im

    def __len__(self):
        return len(self.listing)

input_images_dir = 'input/train/'
label_images_dir = 'input/labels/'
num_epochs=5
learning_rate=0.1
train_on = 5 
val_on = 3
test_on = 2
batch_size= 1

NUM_TRAIN = 5
NUM_VAL = 3
NUM_TEST = 2

listing = random.sample(os.listdir(input_images_dir),5)
train_listing = listing[:min(NUM_TRAIN,train_on)]
val_listing = listing[NUM_TRAIN:min(NUM_VAL+NUM_TRAIN,val_on+NUM_TRAIN)]
test_listing = listing[NUM_VAL+NUM_TRAIN:min(NUM_VAL+NUM_TRAIN+NUM_TEST,test_on+NUM_VAL+NUM_TRAIN)]

data_dir = (input_images_dir,label_images_dir)

input_transform = transforms.Compose([flow_transforms.ArrayToTensor()])
labels_transform = transforms.Compose([flow_transforms.ArrayToTensor()])
#target_labels_transform = transforms.Compose([flow_transforms.ArrayToTensor()])

train_dataset = ListDataset(data_dir,train_listing,input_transform,labels_transform)

val_dataset = ListDataset(data_dir,val_listing,input_transform,labels_transform)

test_dataset = ListDataset(data_dir,test_listing,input_transform,labels_transform)

print("Loading data...")
train_loader = data_utils.DataLoader(train_dataset,batch_size,shuffle = True, drop_last=True)
val_loader = data_utils.DataLoader(val_dataset,batch_size,shuffle = True, drop_last=True)
test_loader = data_utils.DataLoader(test_dataset,batch_size,shuffle = True, drop_last=True)
print("Data loaded!!")

dtype = torch.FloatTensor
model = Model()
model.type(dtype)

loss_fn = torch.nn.MSELoss()

def run_epoch(model, loss_fn, loader, optimizer, dtype):
    """
    Train the model for one epoch.
    """
    running_loss = 0
    count = 0
    for x,y in loader:
        # Inputs
        x_var = Variable(x.type(dtype)) # shape = 1x3x2710x3384 (N,C,H,W) Floattensor
        # Labels
        y_var = Variable(y.type(dtype)) # shape = 1x1x2710x3384, Floattensor

        # Prediction
        pred_labels = model(x_var) # shape = 1x1x2710x3384, FloatTensor
        
        # Loss
        loss = loss_fn(pred_labels, y_var)
        running_loss += loss.data.cpu().numpy()
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_loss/count


for epoch in range(num_epochs):
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    loss = run_epoch(model, loss_fn, train_loader, optimizer,dtype)
    print('Loss for epoch {} : {}'.format(epoch+1,loss))



