# Face-Recognition
Face Recognition by ArcFace. **Accuracy: 97.6%.**

## 1 Dataset prepare
we used **105_classes_pins_dataset** dataset to finetune, and following step must be done before training. </br>
Dataset download(including test dataset and label): 链接：https://pan.baidu.com/s/1jKC-MGQykyV078Mqpy51Ug  提取码：ltf7

### 1.1 Split Train and Validation
we used 4:1 proportion to split **105_classes_pins_dataset**, and you should change the **path** in `split_train_val.py`.

```
cd ./datasets
python split_train_val.py
```

### 1.2 Prepare pair.txt
In this term, we divided the validation set into 2000+ matched pairs, and it including half positive and negative. This pair will be use in evaluation. </br>
You should change the **INPUT_DATA** in `pair_txt.py`.

```
cd ./datasets
python pair_txt.py
```

## 2 Models
Backbone has two selection: FaceMobileNet and Resnet. </br>
Metric: ArcFace. </br>
Loss Function: FocalLoss. 

## 3 Train
In this term, we will finetune the model, and get the best model as well as it's threshold.

```
# train by single gpu
python train_single_gpu.py
```

```
# train by multi gpu
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_gpu.py
```

## 4 Predict
In this term, threshold is your best model's threshold.

```
python predict.py
```

## 5 Config
Don't forget to modify `config.py`, you should modify it according to your environment.

## 6 Pretrained Model Release
Resnet: 链接：https://pan.baidu.com/s/1e3pellia31CsFhkvzFMm9w  提取码：ltf7 </br>
FaceMobileNet: 链接：https://pan.baidu.com/s/1kgA3d7ZvhA4Etb-BbSYFCA  提取码：ltf7
