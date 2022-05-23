## 2022 AI CUP CropClassification

### Setup

Python 3.7
Torch  1.10

Pretrained model:

Swin-Transformer-L Official Github

224x224

https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth

384x384

https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth

ConvNeXt Official Github

https://github.com/facebookresearch/ConvNeXt

### How to train.py

```python
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5) #5e-4
    parser.add_argument('--wd', type=float, default=1e-8) #5e-2

    parser.add_argument('--data-path', type=str,
                        default="./data/train")


    parser.add_argument('--weights', type=str, default='./weight/convnext_xlarge_22k_224.pth',
                        help='initial weights path') #預訓練模型位置
    
    parser.add_argument('--freeze-layers', type=bool, default=False) #freeze = True -> 只訓練head
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
```
run train.py

### How to predict.py

```python
	test_path  # test data 格式
	model_list # model號碼
	model_name # model命名格式
	
```
run predict.py 
