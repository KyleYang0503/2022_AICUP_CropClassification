# -*- coding: utf-8 -*-
"""
Created on Sun May 15 11:02:30 2022

@author: Kyle
"""

import pandas as pd
import torch
import json
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


ensemble_list = [
                './result/swin_384_model_8_result.csv',
                 './result/swin_384_model_9_result.csv',
                 './result/swin_384_model_10_result.csv',
                 './result/swin_384_model_11_result.csv',
                 './result/swin_384_model_12_result.csv',
                 './result/swin_384_model_13_result.csv',
                 './result/swin_384_model_14_result.csv',
                 './result/swin_384_model_15_result.csv',
                 './result/swin_384_model_16_result.csv',
                 './result/swin_384_model_17_result.csv',
                 './result/conv_224_epoch_15_result.csv',
                 './result/conv_224_epoch_15_result.csv',
                 './result/conv_224_epoch_20_result.csv',
                 './result/conv_224_epoch_23_result.csv',
                 './result/conv_224_epoch_24_result.csv',
                 './result/conv_224_epoch_25_result.csv',
                 './result/conv_224_epoch_26_result.csv',
                 './result/conv_224_epoch_27_result.csv']

columes = {'image_filename','0','1','2','3','4','5','6'
                                     ,'7','8','9','10','11','12','13'}
index = {
    "banana":0,
    "bareland":1,
    "carrot":2,
    "corn":3,
    "dragonfruit":4,
    "garlic":5,
    "guava":6,
    "peanut":7,
    "pineapple":8,
    "pumpkin":9,
    "rice":10,
    "soybean":11,
    "sugarcane":12,
    "tomato":13
}

json_path = './class_indices.json'
json_file = open(json_path, "r")
class_indict = json.load(json_file)

result = pd.DataFrame(columns=columes)
df_ = pd.read_csv(ensemble_list[0],index_col="image_filename")
dict_2d = {}

for x in list(df_.index.values):
    dict_2d.update({x : [0,0,0,0,0,0,0,0,0,0,0,0,0,0]})

for z,item in enumerate(ensemble_list):
    df = pd.read_csv(item,index_col="image_filename")
    df = df.sort_index(axis=0)
    for x in list(df.index.values):
        dict_2d[x][index[df.loc[x].values[0]]] += 1.

i=0
data = pd.DataFrame(columns=['image_filename','label'])

for x in list(df_.index.values):
    raw = torch.from_numpy(np.array(dict_2d[x]))
    output = torch.squeeze(raw).cpu()
    predict = torch.softmax(output, dim=0)

    predict_cla = torch.argmax(predict).numpy()
    i+=1
    data = data.append({'image_filename':x,'label':class_indict[str(predict_cla)]},
                      ignore_index=True)
    print('\r%d'%i,end='')

ensemble_name = 's7_diff_all'

data.to_csv("./result/%s_result.csv"%ensemble_name,index=False)
    
