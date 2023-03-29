import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader

from src.utils import seed_everything, EarlyStopping
from src.model import *
from src.kd_loss.st import SoftTargetLoss
from src.kd_loss.fitnet import HintLearningLoss
from src.kd_loss.mbd import DeltaMeritBasedLoss, HardMeritBasedLoss, SoftMeritBasedLoss

parser = argparse.ArgumentParser(description='run knowledge distillation')

parser.add_argument('--model_type', default='normal', help='if you want to use resnet, you type resnet')
parser.add_argument('--epoch', type=int, default=100, help='number of total epochs to run')
parser.add_argument('--batch_size',type=int, default=128, help='the size of batch')
parser.add_argument('--num_class', type=int, default=10, choices=[10, 100], help='number of classes')
parser.add_argument('--kd', choices=['st', 'fitnet', 'takd', 'multi', 'mbd-delta', 'mbd-hard', 'mbd-soft'], default='st', help='define distillation method')

args = parser.parse_args()

def main():
    seed_everything(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = args.epoch 
    batch_size = args.batch_size
    num_class = args.num_class
    model_type = args.model_type
    kd = args.kd
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
        teacher = TeacherModel(output_dim=num_class).to(device)
        ta = TeacherAssistantModel(output_dim=num_class).to(device)
        student = StudentModel(output_dim=num_class).to(device)
        regressor_ta = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)).to(device)
        regressor_student = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True)).to(device)
        regressor = nn.Sequential(nn.Conv2d(32, 128, 3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True)).to(device)
    else:
        teacher = resnet50(output_dim=num_class).to(device)
        ta = resnet_assistant(output_dim=num_class).to(device)
        student = resnet_student(output_dim=num_class).to(device)
        regressor_ta = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True)).to(device)
        regressor_student = nn.Sequential(nn.Conv2d(128,256,3,padding=1),
                                          nn.BatchNorm2d(512), 
                                          nn.ReLU(inplace=True)).to(device)
        regressor = nn.Sequential(nn.Conv2d(128,512,3,padding=1),
                                  nn.BatchNorm2d(512), 
                                  nn.ReLU(inplace=True)).to(device)
    
    
    # setting loss function
    if kd == 'st':
        loss_fn = nn.CrossEntropyLoss()
        kd_loss_fn = SoftTargetLoss()
        optimizer = Adam(student.parameters())
    elif kd == 'fitnet':
        hint_loss_fn = HintLearningLoss()
        loss_fn = nn.CrossEntropyLoss()
        kd_loss_fn = SoftTargetLoss()
        optimizer_hint_student = Adam(list(student.parameters()) + list(regressor.parameters()))
        optimizer = Adam(student.parameters())
    elif kd == 'takd':
        loss_fn = nn.CrossEntropyLoss()
        kd_loss_fn = SoftTargetLoss()
        optimizer_student = Adam(student.parameters())
        optimizer_ta = Adam(ta.parameters())
    elif kd == 'multi':
        hint_loss_fn = HintLearningLoss()
        loss_fn = nn.CrossEntropyLoss()
        kd_loss_fn = SoftTargetLoss()
        optimizer_hint_ta = Adam(list(ta.parameters()) + list(regressor_ta.parameters()))
        optimizer_ta = Adam(ta.parameters())
        optimizer_hint_student = Adam(list(student.parameters()) + list(regressor_student.parameters()))
        optimizer_student = Adam(student.parameters())
    elif kd == 'mbd-delta':
        loss_fn = nn.CrossEntropyLoss()
        kd_loss_fn = DeltaMeritBasedLoss()
        optimizer = Adam(student.parameters())
    elif kd == 'mbd-hard':
        loss_fn = nn.CrossEntropyLoss()
        kd_loss_fn = HardMeritBasedLoss()
        optimizer = Adam(student.parameters())
    elif kd == 'mbd-soft':
        loss_fn = nn.CrossEntropyLoss()
        kd_loss_fn = SoftMeritBasedLoss()
        optimizer = Adam(student.parameters())
    
    # train step
    if kd == 'st':
        print('Distillation of Hinton')
        print('teacher: teacher, student: student')
        teacher.load_state_dict(torch.load('./logs/teacher/' +model_type + '_' + str(epoch) + '_' + str(batch_size) +'_' + str(num_class) +'_param.pth'))
        for t in range(epoch):
            train_acc, train_loss = train_kd(student, teacher, optimizer, train_dataloader, kd_loss_fn, device)
            val_acc, val_loss = valid(student, val_dataloader, loss_fn,  device)
            print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} |")
    
    elif kd == 'takd':
        teacher.load_state_dict(torch.load('./logs/teacher/' +model_type + '_' + str(epoch) + '_' + str(batch_size) +'_' + str(num_class) +'_param.pth'))
        print('Distillation by TAKD')
        print('1st    teacher: teacher, student: ta')
        for t in range(epoch):
            train_acc, train_loss = train_kd(ta, teacher, optimizer_ta, train_dataloader, kd_loss_fn, device)
            val_acc, val_loss = valid(ta, val_dataloader, loss_fn,  device)
            print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} |")
        print('--------------------------------------------')
        print('2nd    teacher: ta, student: student')
        for t in range(epoch):
            train_acc, train_loss = train_kd(student, ta, optimizer_student, train_dataloader, kd_loss_fn, device)
            val_acc, val_loss = valid(student, val_dataloader, loss_fn,  device)
            print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} |")
    
    elif kd == 'fitnet':
        teacher.load_state_dict(torch.load('./logs/teacher/' +model_type + '_' + str(epoch) + '_' + str(batch_size) +'_' + str(num_class) +'_param.pth'))
        print('teacher: teacher, student: student')
        print('1st    Hint learning')
        for t in range(epoch):
            hint_loss = hint_training(student, teacher, regressor, optimizer_hint_student, train_dataloader, hint_loss_fn, device)
            hint_val_loss = hint_validation(student, teacher, regressor, val_dataloader, hint_loss_fn, device)
            print(f"| {t:>6} |  {round(hint_loss, 7):>12} | {round(hint_val_loss, 7):>12} |")
        print('--------------------------------------------')
        print('2nd    Distillation')
        for t in range(epoch):
            train_acc, train_loss = train_kd(student, teacher, optimizer, train_dataloader, kd_loss_fn, device)
            val_acc, val_loss = valid(student, val_dataloader, loss_fn,  device)
            print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} |")
    
    elif kd == 'multi':
        teacher.load_state_dict(torch.load('./logs/teacher/' +model_type + '_' + str(epoch) + '_' + str(batch_size) +'_' + str(num_class) +'_param.pth'))
        print('teacher: teacher, student: ta')
        print('1st    Hint learning')
        for t in range(epoch):
            hint_loss = hint_training(ta, teacher, regressor_ta, optimizer_hint_ta, train_dataloader, hint_loss_fn, device)
            hint_val_loss = hint_validation(ta, teacher, regressor_ta, val_dataloader, hint_loss_fn, device)
            print(f"| {t:>6} |  {round(hint_loss, 7):>12} | {round(hint_val_loss, 7):>12} |")
        print('--------------------------------------------')
        print('2nd    Distillation')
        for t in range(epoch):
            train_acc, train_loss = train_kd(ta, teacher, optimizer_ta, train_dataloader, kd_loss_fn, device)
            val_acc, val_loss = valid(ta, val_dataloader, loss_fn,  device)
            print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} |")
        print('teacher: ta, student: student')
        print('1st    Hint learning')
        for t in range(epoch):
            hint_loss = hint_training(student, ta, regressor_student, optimizer_hint_student, train_dataloader, hint_loss_fn, device)
            hint_val_loss = hint_validation(student, ta, regressor_student, val_dataloader, hint_loss_fn, device)
            print(f"| {t:>6} |  {round(hint_loss, 7):>12} | {round(hint_val_loss, 7):>12} |")
        print('--------------------------------------------')
        print('2nd    Distillation')
        for t in range(epoch):    
            train_acc, train_loss = train_kd(student, ta, optimizer_student, train_dataloader, kd_loss_fn, device)
            val_acc, val_loss = valid(ta, val_dataloader, loss_fn,  device)
            print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} |")
    
    if kd == 'mbd-delta':
        print('Delta Merit-basd Distillation')
        print('teacher: teacher, student: student')
        teacher.load_state_dict(torch.load('./logs/teacher/' +model_type + '_' + str(epoch) + '_' + str(batch_size) +'_' + str(num_class) +'_param.pth'))
        for t in range(epoch):
            train_acc, train_loss = train_kd(student, teacher, optimizer, train_dataloader, kd_loss_fn, device)
            val_acc, val_loss = valid(student, val_dataloader, loss_fn,  device)
            print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} |")
    
    if kd == 'mbd-soft':
        print('Soft Merit-basd Distillation')
        print('teacher: teacher, student: student')
        teacher.load_state_dict(torch.load('./logs/teacher/' +model_type + '_' + str(epoch) + '_' + str(batch_size) +'_' + str(num_class) +'_param.pth'))
        for t in range(epoch):
            train_acc, train_loss = train_kd(student, teacher, optimizer, train_dataloader, kd_loss_fn, device)
            val_acc, val_loss = valid(student, val_dataloader, loss_fn,  device)
            print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} |")
    
    if kd == 'mbd-hard':
        print('Hard Merit-basd Distillation')
        print('teacher: teacher, student: student')
        teacher.load_state_dict(torch.load('./logs/teacher/' +model_type + '_' + str(epoch) + '_' + str(batch_size) +'_' + str(num_class) +'_param.pth'))
        for t in range(epoch):
            train_acc, train_loss = train_kd(student, teacher, optimizer, train_dataloader, kd_loss_fn, device)
            val_acc, val_loss = valid(student, val_dataloader, loss_fn,  device)
            print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} |")
    
    
def hint_training(student, teacher, regressor, optimizer, dataloader, loss_fn, device):
    student.train()
    teacher.eval()
    regressor.train()
    avg_loss = 0
    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            middle_targets = teacher.extract_features(images)
        middle_logits = student.extract_features(images)
        guide_logits = regressor(middle_logits)
        loss = loss_fn(guide_logits, middle_targets)
        avg_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_loss /= len(dataloader)
    return avg_loss

def hint_validation(student, teacher, regressor, dataloader, loss_fn, device):
    student.eval()
    teacher.eval()
    regressor.eval()
    avg_loss = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                middle_targets = teacher.extract_features(images)
            middle_logits = student.extract_features(images)
            guide_logits = regressor(middle_logits)
            loss = loss_fn(guide_logits, middle_targets)
            avg_loss += loss.item()
            
        avg_loss /= len(dataloader)
        return avg_loss

def train_kd(student, teacher, optimizer, dataloader, loss_fn, device):
    student.train()
    teacher.eval()
    avg_acc = 0
    avg_loss = 0
    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = student(images)
        targets = teacher(images)
        loss = loss_fn(logits, targets, labels)
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
        for images, labels in tqdm(dataloader, leave=False):
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