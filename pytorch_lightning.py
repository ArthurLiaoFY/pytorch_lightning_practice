# %%
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning.pytorch as pl
data_dir = r'C:\Users\USER\OneDrive\文件\Python Scripts\flower_data'
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import wandb

import multiprocessing
print('CPU count : ', multiprocessing.cpu_count())
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.models import VGG16_Weights

import cv2
from PIL import Image
from torchvision.models import ResNeXt50_32X4D_Weights
from torch.utils.data import DataLoader


# wandb.init(
#     # set the wandb project where this run will be logged
#     project="flower_data_project",
    
#     # track hyperparameters and run metadata
#     config={
#     "architecture": "VGG16",
#     "epochs": 100,
#     }
# )
# %%        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%
data_dir = 'C:\\Users\\USER\\OneDrive\\文件\\Python Scripts\\flower_data'
# %%
config = {
    'batch': 32,
    'epochs': 100,
    'n_class': 102,
    'vgg_resize_shape': 224,
    'vgg_mean_transform':[0.48235, 0.45882, 0.40784],
    'vgg_std_transform':[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
}
# %%
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((config['vgg_resize_shape'],config['vgg_resize_shape'])),
        ####################
        # augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),        
        transforms.RandomRotation(30,),
        transforms.RandomCrop(config['vgg_resize_shape']),
        ####################
        transforms.ToTensor(),
        transforms.Normalize(mean=config['vgg_mean_transform'], std=config['vgg_std_transform']),
    ]),
    'valid': transforms.Compose([
        transforms.Resize((config['vgg_resize_shape'],config['vgg_resize_shape'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['vgg_mean_transform'], std=config['vgg_std_transform']),
    ]),
    'test': transforms.Compose([
        transforms.Resize((config['resize_shape'],config['resize_shape'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['vgg_mean_transform'], std=config['vgg_std_transform']),
    ]),
}
# %%

class BirdDataset(Dataset):
    def __init__(self, dataset, img_transform):
        self.img_transform = img_transform
        self.dataset = dataset
        datas = []
        assert dataset in ('train', 'valid')
        for root, labels, img_paths in os.walk(os.path.join(os.getcwd(), self.dataset)):
            for label in labels:
                for sub_root, _, img_paths in os.walk(os.path.join(root, str(label))):
                    for img_path in img_paths:
                        datas.append([int(label), os.path.join(sub_root, img_path)])

        self.config = pd.DataFrame(datas, columns=['label', 'img_path'])
        self.img_label = self.config['label']
        self.img_path = self.config['img_path']


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        # should divide by 255
        image = Image.open(self.img_path[idx])
        # should not change the label
        label = torch.tensor(self.img_label[idx], dtype=torch.long) - 1
        image = self.img_transform[self.dataset](image)

        return image, label
        
# %%
class BirdDatasetTest(Dataset):
    def __init__(self, dataset, img_transform):
        self.img_transform = img_transform
        self.dataset = dataset
        datas = []
        assert dataset == 'test'
        for root, _, img_paths in os.walk(os.path.join(os.getcwd(), self.dataset)):
            for img_path in img_paths:
                datas.append([os.path.join(root, img_path)])

        self.config = pd.DataFrame(datas, columns=['img_path'])
        self.img_path = self.config['img_path']


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        # should divide by 255
        image = Image.open(self.img_path[idx])
        image = self.img_transform[self.dataset](image)
        
        return image

# %%
train_dataset = BirdDataset(dataset='train', img_transform=data_transforms)
valid_dataset = BirdDataset(dataset='valid', img_transform=data_transforms)
test_dataset = BirdDatasetTest(dataset='test', img_transform=data_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=config['batch'], shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch'], shuffle=False, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch'], shuffle=False, pin_memory=True)

dataset_sizes={}
dataset_sizes['train'] = len(train_dataset.img_path)
dataset_sizes['valid'] = len(valid_dataset.img_path)
dataset_sizes['test'] = len(test_dataset.img_path)
# os.environ['WANDB_API_KEY'] = '0782b0393bbac75f7c807f3586ea3c56fc52ca53'
# %%


cross_entropy = nn.CrossEntropyLoss()
class VGGModel(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.VGG = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # freeze some parameters from training
        for pram in VGG.features.parameters():
            pram.requires_grad = False
        # keep classifier parameters un-freeze

    def forward(self, x):
        return torch.softmax(
            self.VGG(x),
            dim=0
        )
    
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.n_class = 102
        self.VGG = VGGModel(self.n_class)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)



# %%


trainer = pl.Trainer(max_epochs=100)
model = LitModel()

# trainer.fit(model, train_dataloaders=train_dataloader)
# trainer.validate(model, train_dataloaders=valid_dataloader)
# trainer.test(model, train_dataloaders=test_dataloader)

# %%
a, b = next(iter(train_dataloader))
print(a.shape)
print(b.shape)
# %%
