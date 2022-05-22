# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:55:16 2022

@author: Kyle
"""

import pandas as pd
import torch
import json

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


ensemble_list = ['./result/swin_384_model_8_ens.csv',
                 './result/swin_384_model_9_ens.csv',
                 './result/swin_384_model_10_ens.csv',
                 './result/swin_384_model_11_ens.csv',
                 './result/swin_384_model_12_ens.csv',
                 './result/swin_384_model_13_ens.csv',
                 './result/swin_384_model_14_ens.csv',
                 './result/swin_384_model_15_ens.csv',
                 './result/swin_384_model_16_ens.csv',
                 './result/swin_384_model_17_ens.csv']
columes = {'img':'','0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0
                                     ,'7':0,'8':0,'9':0,'10':0,'11':0,'12':0,'13':0}
result = pd.read_csv(ensemble_list[0],index_col="img")
result = result.sort_index(axis=0)

json_path = './class_indices.json'
json_file = open(json_path, "r")
class_indict = json.load(json_file)


for z,item in enumerate(ensemble_list):
    if z==0:
        continue
    df = pd.read_csv(item,index_col="img")
    df = df.sort_index(axis=0)
    if (list(df.index.values) == list(result.index.values)):
        result = result.add(df)
    else:
        print("%d index error")%z

#result = result.div(len(ensemble_list))

data = pd.DataFrame(columns=['image_filename','label'])
i=0
for item in list(result.index.values):
    
    raw = torch.from_numpy(result.loc[item].values)
    output = torch.squeeze(raw).cpu()
    predict = torch.softmax(output, dim=0)

    predict_cla = torch.argmax(predict).numpy()
    i+=1
    print("\r%d"%i,end='')

    data = data.append({'image_filename':item,'label':class_indict[str(predict_cla)]},
                      ignore_index=True)

ensemble_name = 's10'

data.to_csv("./result/%s_result.csv"%ensemble_name,index=False)