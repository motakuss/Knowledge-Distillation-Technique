from calendar import EPOCH
import os
import argparse
import logging
from pyexpat import model
from random import seed
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader


from src.utils import save_param, seed_everything, EarlyStopping, get_lr, save_param
from src.model import *

parser = argparse.ArgumentParser(description='run knowledge distillation')

parser.add_argument('--model_type', default='normal', choices=['normal', 'resnet'], help='if you want to use [resnet], you type resnet')
parser.add_argument('--model_size', default='teacher', choices=['teacher', 'ta', 'student'], help='selsect model teacher or ta or student')
parser.add_argument('--epoch', type=int, default=100, help='number of total epochs to run')
parser.add_argument('--batch_size',type=int, default=128, help='the size of batch')
parser.add_argument('--num_class', type=int, default=10, choices=[10, 100], help='number of classes')

args = parser.parse_args()

def main():
    seed_everything(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = args.epoch 
    batch_size = args.batch_size
    num_class = args.num_class
    model_type = args.model_type
    model_size = args.model_size
    es = EarlyStopping(patience=5, verbose=1)
    
    # setting dataset(cifar10 or cifar100)
    if num_class == 10:
        data_dir = './data/cifar10'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        trainset = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=transform)
        testset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transform)
    else:
        data_dir = './data/cifar100'
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean= [0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
        trainset = datasets.CIFAR100(root=data_dir,download=True,train=True,transform=transform)
        testset = datasets.CIFAR100(root=data_dir,download=True,train=False,transform=transform)
    n_train = int(len(trainset) * 0.8)
    n_val = len(trainset) - n_train
    trainset, valset = random_split(trainset, [n_train, n_val])
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # setting model
    if model_type == 'normal':
        if model_size == 'teacher':
            model = TeacherModel(output_dim=num_class).to(device)
        elif model_size == 'ta':
            model = TeacherAssistantModel(output_dim=num_class).to(device)
        elif model_size == 'student':
            model = StudentModel(output_dim=num_class).to(device)
    elif model_type == 'resnet':
        if model_size == 'teacher':
            model = resnet50(output_dim=num_class).to(device)
        elif model_size == 'ta':
            model = resnet_assistant(output_dim=num_class).to(device)
        elif model_size == 'student':
            model = resnet_student(output_dim=num_class).to(device)
    
    # setting loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, [10, 15], gamma=0.1)
    
    # train
    for t in range(epoch):
        lr = get_lr(optimizer)
        train_acc, train_loss = train(model, optimizer, train_dataloader, loss_fn, device)
        val_acc, val_loss = valid(model, val_dataloader, loss_fn,  device)
        print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} | {round(lr, 7):>8} |")
        scheduler.step()
        if save_param(es, val_loss, model, model_type, model_size, epoch, batch_size, num_class):
            break
    
def train(model, optimizer, dataloader, loss_fn, device):
    model.train()
    avg_acc = 0
    avg_loss = 0
    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = loss_fn(logits, labels)
        avg_loss += loss.item()
        
        preds = logits.argmax(dim=1, keepdim=True)
        avg_acc += preds.eq(labels.view_as(preds)).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_acc = 100. * avg_acc / len(dataloader.dataset)
    avg_loss /= len(dataloader)
    return avg_acc, avg_loss

def valid(model, dataloader, loss_fn, device):
    model.eval()
    avg_acc = 0
    avg_loss = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = loss_fn(logits, labels)
            avg_loss += loss.item()
            
            preds = logits.argmax(dim=1, keepdim=True)
            avg_acc += preds.eq(labels.view_as(preds)).sum().item()
            
    avg_acc = 100. * avg_acc / len(dataloader.dataset)
    avg_loss /= len(dataloader)
    return avg_acc, avg_loss


if __name__ == '__main__':
    main()