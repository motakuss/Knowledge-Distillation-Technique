import torch
import os
import random
import numpy as np

class EarlyStopping:
    def __init__(self, patience, verbose):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose
    
    def __call__(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss
        return False
    
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def save_param(es, loss, model, model_type, model_size, epoch, batch_size, num_class):
    if es(loss):
        return True
    else:
        torch.save(model.state_dict(), './logs/' + model_size + '/' +model_type + '_' + str(epoch) + '_' + str(batch_size) +'_' + str(num_class) +'_param.pth')
        return False 

