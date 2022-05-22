# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:07:40 2022

@author: Kyle
"""

import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from model import swin_large_patch4_window12_384_in22k as create_model

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


def estimate(img,predict,df,map_,real):
    label = df[df['img']==img]['label'].values[0]
    if predict == label:
        #print("Correct\n")
        map_[str(label)] +=1
        real += 1
    #else:
        #print("Wrong,predict:%s but correct label is : %s"%(predict,label))
    return map_,real
def get_class_len(df,class_indict,num_classes):
    class_len =[]
    for i in range (num_classes):
        class_len.append(len(df[df['label'] == class_indict[str(i)]]))
    return class_len

def class_accuracy(class_len,map_):
    class_accuracy_map = map_.copy()
    i=0
    for item in class_accuracy_map:
        class_accuracy_map[item] = "%.2f"%((map_[item]/class_len[i])*100)
        i+=1
    return class_accuracy_map

def test():
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 219
    img_size = 224
    data_transform = transforms.Compose([transforms.Resize([img_size,img_size]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    test_path = "./data/test"
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model = create_model(num_classes=num_classes).to(device)
    
    model_list_= ['0']
    for item in model_list_:
        print("current is model : %s\n"%item)
        model_name = 'model_%s'%item
        ensemble_name = 'swin_224_%s'%model_name
    
        # load model weights
        model_weight_path = "./model/%s.pth"%model_name#./weights/best_model.pth
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        model.cuda()
        
        ensemble = pd.DataFrame(columns=['img','0','1','2','3','4','5','6'
                                         ,'7','8','9','10','11','12','13'])
        result = pd.DataFrame(columns=['image_filename','label'])
        
        i=0
        real = 0
        f = os.listdir(test_path)
        for x in f:
            img_path = test_path + "/" + x
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            #plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
        
            # read class_indict
            # create model
    
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
            #print("img:%s,predict:%s"%(x,class_indict[str(predict_cla)]))
            
            ensemble_output = output.numpy()
            dict_ = {'img':x,'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0
                                         ,'7':0,'8':0,'9':0,'10':0,'11':0,'12':0,'13':0}
            for y in range (14):
                dict_[str(y)] = ensemble_output[y]
            
            ensemble = ensemble.append(dict_,ignore_index=True)
            result = result.append({'image_filename':x,'label':class_indict[str(predict_cla)]},
                          ignore_index=True)
            #map_,real = estimate(x,class_indict[str(predict_cla)],df,map_,real)
            i+=1
            print("\rAmount:%d, Current acc:%.2f"%(i,(real/i)*100),end='')
            
    
        result.to_csv("./result/%s_result.csv"%ensemble_name,index=False)
        ensemble.to_csv("./result/%s_ens.csv"%ensemble_name,index=False)
    '''
    with open('./result/class_acc.json', 'w') as f:
        json.dump(class_accuracy_map, f)
    print(class_accuracy_map)
    '''
if __name__ == '__main__':
    test()

'''
[transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
'''