# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:24:38 2022

@author: Kyle
"""

import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import shutil
from PIL import Image

def read_split_data(root: str, val_rate: float = 0.1):
    random.seed(0)  
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  
    train_images_label = [] 
    val_images_path = []  
    val_images_label = [] 
    every_class_num = [] 
    supported = [".jpg", ".JPG", ".png", ".PNG"]  
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path: 
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))


    return train_images_path, train_images_label, val_images_path, val_images_label




def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  
    accu_num = torch.zeros(1).to(device)   
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   
    accu_loss = torch.zeros(1).to(device) 

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num




def train_by_class(label_path,data_path,target):
    df = pd.read_csv(label_path,index_col = 'filename')
    i=0
    for x in list(df.index.values):
        label = df.loc[x].values[0]
        if not os.path.exists(target +'/%d'%label):
            os.makedirs(target + '/%d'%label)
        shutil.copy(data_path + '/' + x,"./data/train" + '/%d/'%label + x)
        '''
        if i%10==0:
            shutil.copy(data_path + '/' + x,"./data/train_0.9/test" + '/%d/'%label + x)
        else:
            shutil.copy(data_path + '/' + x,"./data/train_0.9/train" + '/%d/'%label + x)
        i+=1
        '''
#train_by_class("./data/label.csv","./data/training",'./data/train')

def rotate_enhance(data_path):
    for class_ in os.listdir(data_path):
        dir_path = data_path + '/%s'%class_
        image_list = os.listdir(dir_path)
        print(class_)
        for item in image_list:
            img = Image.open(dir_path + "/%s"%item)
            angle = 90
            op = [Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270]
            for i in op:
                new_img = img.transpose(i)
                new_img.save(dir_path +"/%s_%d.jpg"%(item,angle))
                angle += 90
            angle = 0
                
#rotate_enhance('./data/train')


    
    
    
    
    
    
