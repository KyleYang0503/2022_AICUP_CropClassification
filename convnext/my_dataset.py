# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:06:28 2022

@author: Kyle
"""

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])

        label = self.images_class[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
    
'''
data_transform =transforms.Compose([transforms.CenterCrop(224),
                                   
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()])
images_path=["./data/train/pumpkin/160112-2-0078.JPG","./data/train/tomato/160107-3-0013.JPG"]
images_class=["1","2"]

d = MyDataSet(images_path=images_path
              ,images_class=images_class,
              transform=data_transform)
s  = d[0][0]
imshow(s)
'''