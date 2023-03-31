# Knowledge-Distillation-Technique

**Knowledge Distillation is a technique for transferring the knowledge of a large deep neural network (DNN) to a smaller neural network.** 

Typically, a large DNN has high predictive accuracy but requires high computational costs and memory usage. On the other hand, smaller neural networks have lower computational costs and memory usage but may have inferior accuracy.

**Knowledge Distillation enables a smaller model to achieve high accuracy by transferring the knowledge of a large DNN to the smaller model.**  
 Specifically, the output of a large DNN is used as the teacher mapping to train a smaller model, transferring the knowledge of the large model to the small model.

This technique enables smaller neural networks with lower computational costs and memory usage to achieve high predictive accuracy, thereby increasing the practicality of machine learning models.

## Papaers

* [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Hinton, 2015
* [FitNets: Hint for Thin Deep Nets](https://arxiv.org/abs/1412.6550), Romero, 2015
* [Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393), Mirzadeh, 2019
* [Multi-Stage Model Compression using Teacher Assistant and Distillation with Hint-Based Training](https://ieeexplore.ieee.org/document/9767229), Morikawa, 2022
* CNN Model Compression by Merit-based Distillation, Morikawa, 2023

## Setup
If you use venv :
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
## Run
The case of train.py :  
This program is used to obtain the results of the baseline.
```
python train.py --model_type normal --model_size teacher --epoch 100 --batch_size 128 --num_class
```
* --model_type -> [normal, resnet]
    - If "normal" is selected, a CNN type model is set, and if "resnet" is selected, a resNet type model is set.
* --model_size -> [teacher, ta, student]
    - The model of the selected size will be set. Please refer to the Model table for details.
* --epoch -> [default 100]
    - Please set it freely.
* --batch_size -> [default 128]
    - Please set it freely.
* --num_class -> [10, 100]
    - If 10 is selected, the training will be conducted using CIFAR-10, and if 100 is selected, the training will be conducted using CIFAR-100.

The case of train_kd.py :   
This program is used to obtain the results of the using distillation method.
```
python train_kd.py --model_type normal --epoch 100 --batch_size 128 --num_class 10 --kd st
```
* --model_type -> [normal, resnet]
    - If "normal" is selected, a CNN type model is set, and if "resnet" is selected, a resNet type model is set.
* --epoch -> [default 100]
    - Please set it freely.
* --batch_size -> [default 128]
    - Please set it freely.
* --num_class -> [10, 100]
    - If 10 is selected, the training will be conducted using CIFAR-10, and if 100 is selected, the training will be conducted using CIFAR-100.
* --kd -> [st, fitnet, takd, multi, mbd-delta, mbd-hard, mbd-soft]
    - Choose the distillation method. Please refer to the Lists for details.

## Lists

|    Name    |    Method   |     Paper Link    |     Code Link    |
|:-----------|------------:|:------------:|:------------:|
|BaseLine| -| -|[train.py](https://github.com/motakuss/Knowledge-Distillation-Technique/blob/main/train.py)|
|st|soft target|[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)|[st.py](https://github.com/motakuss/Knowledge-Distillation-Technique/blob/main/src/kd_loss/st.py)|
|fitnet|Hint-training|[FitNets: Hint for Thin Deep Nets](https://arxiv.org/abs/1412.6550)|[fitnet.py](https://github.com/motakuss/Knowledge-Distillation-Technique/blob/main/src/kd_loss/fitnet.py)|
|takd|Teacher Assistant Knowledge Distillation|[Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393)|[train_kd.py](https://github.com/motakuss/Knowledge-Distillation-Technique/blob/main/train_kd.py)|
|multi|Multi Hint-training & Distillation|[Multi-Stage Model Compression using Teacher Assistant and Distillation with Hint-Based Training](https://ieeexplore.ieee.org/document/9767229)|[train_kd.py](https://github.com/motakuss/Knowledge-Distillation-Technique/blob/main/train_kd.py)|
|mbd-delta|Delta Merit-based Distillation|CNN Model Compression by Merit-based Distillation|[mbd.py](https://github.com/motakuss/Knowledge-Distillation-Technique/blob/main/src/kd_loss/mbd.py)|
|mbd-hard|Hard Merit-based Distillation|CNN Model Compression by Merit-based Distillation|[mbd.py](https://github.com/motakuss/Knowledge-Distillation-Technique/blob/main/src/kd_loss/mbd.py)|
|mbd-soft|Soft Merit-based Distillation|CNN Model Compression by Merit-based Distillation|[mbd.py](https://github.com/motakuss/Knowledge-Distillation-Technique/blob/main/src/kd_loss/mbd.py)|

## Dataset
Cifar10  
Cifar100

When running train.py for the first time, data download is performed. For cifar10, set the argument to --num_classs 10, and for cifar100, set it to --num_classs 100.

## Model
|    Model    |    Number of Layer   |    Parameter     |     The Role of Model    |
|:-----------|------------:|:------------:|:------------:|
|    CNN8    |    8   |    4,760,010     |     Teacher   |
|    CNN12    |    12   |    1,996,042     |     TA   |
|    CNN16    |    16   |    707,322     |     Student   |
|    ResNet50    |    50   |    23,520,842     |     Teacher   |
|    ResNet71    |    71   |    8,015,018     |     TA   |
|    ResNet98    |    98   |    2,831,578     |     Student   |