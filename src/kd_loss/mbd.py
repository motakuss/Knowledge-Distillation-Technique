import torch
import torch.nn as nn

class DeltaMeritBasedLoss(nn.Module):
    '''
    Merit-based Distillation
    '''
    def __init__(self, T=1):
        super().__init__()
        self.T = T
    
    def forward(self, logits, targets, t, alpha=0.5, delta=0.1):
        '''
        logits : output of the student
        targets: output of the teacher
        t      : correct label
        '''
        logits = logits / self.T
        targets = targets / self.T
        alpha1 = alpha + delta # weight for correct inference
        alpha0 = alpha - delta # weight for incorrect inference
        soft_loss = nn.KLDivLoss(reduction='batchmean')
        hard_loss = nn.CrossEntropyLoss()
        p = nn.Softmax(dim=1)
        q = nn.LogSoftmax(dim=1)
        loss = 0.
        # Checking whether the teacher's inferences are correct
        score = []
        for i in range(len(targets)):
            if torch.argmax(targets[i],dim=-1) == t[i]:
                score.append(1)
            else:
                score.append(0)
        # Calculate loss per inference
        for i in range(len(logits)):
            re_logits = logits[i].unsqueeze(0)
            re_targets = targets[i].unsqueeze(0)
            re_t = t[i].unsqueeze(0)
            KLD_loss = soft_loss(q(re_logits / self.T), p(re_targets / self.T))
            CE_loss = hard_loss(re_logits, re_t)
            # Change the weighting for each inference
            if score[i] == 1:
                loss += (1-alpha1) * CE_loss + alpha1 * self.T * self.T * KLD_loss
            else:
                loss += (1-alpha0) * CE_loss + alpha0 * self.T * self.T * KLD_loss       
        loss /= len(t)
        return loss

class HardMeritBasedLoss(nn.Module):
    def __init__(self, T=1):
        super().__init__()
        self.T = T
    
    def forward(self, logits, targets, t):
        '''
        logits : output of the student
        targets: output of the teacher
        t      : correct label
        '''
        logits = logits / self.T
        targets = targets / self.T
        
        soft_loss = nn.KLDivLoss(reduction='batchmean')
        hard_loss = nn.CrossEntropyLoss()
        p = nn.Softmax(dim=1)
        q = nn.LogSoftmax(dim=1)
        loss = 0.
        # Checking whether the teacher's inferences are correct
        score = []
        for i in range(len(targets)):
            if torch.argmax(targets[i],dim=-1) == t[i]:
                score.append(1)
            else:
                score.append(0)
        # Calculate loss per inference
        for i in range(len(logits)):
            re_logits = logits[i].unsqueeze(0)
            re_targets = targets[i].unsqueeze(0)
            re_t = t[i].unsqueeze(0)
            KLD_loss = soft_loss(q(re_logits / self.T), p(re_targets / self.T))
            CE_loss = hard_loss(re_logits, re_t)
            # Change the weighting for each inference
            if score[i] == 1:
                alpha = 0.5
                loss += (1-alpha) * CE_loss + alpha * self.T * self.T * KLD_loss
            else:
                alpha = 1.0
                loss += alpha * CE_loss + (1-alpha) * self.T * self.T * KLD_loss                
        loss /= len(t)
        return loss

class SoftMeritBasedLoss(nn.Module):
    def __init__(self, T=1):
        super().__init__()
        self.T = T
        
    def forward(self, logits, targets, t):
        '''
        logits : output of the student
        targets: output of the teacher
        t      : correct label
        '''
        logits = logits / self.T
        targets = targets / self.T
        
        soft_loss = nn.KLDivLoss(reduction='batchmean')
        hard_loss = nn.CrossEntropyLoss()
        p = nn.Softmax(dim=1)
        q = nn.LogSoftmax(dim=1)
        loss = 0.
        # Calculate loss per inference
        for i in range(len(logits)):
            re_logits = logits[i].unsqueeze(0)
            re_targets = targets[i].unsqueeze(0)
            re_t = t[i].unsqueeze(0)
            KLD_loss = soft_loss(q(re_logits / self.T), p(re_targets / self.T))
            CE_loss = hard_loss(re_logits, re_t)
            alpha = p(re_targets).squeeze(0)[re_t.squeeze(0)]
            loss += (1-alpha) * CE_loss + alpha * self.T * self.T * KLD_loss
        loss /= len(t)
        return loss
    
if __name__ == '__main__':
    logits = torch.randn((32,10))
    targets = torch.randn((32,10))
    t = torch.randint(0, 10, (32,))
    loss = DeltaMeritBasedLoss()
    kd_loss = loss(logits,targets,t)
    print(kd_loss)
    
    loss = HardMeritBasedLoss()
    kd_loss = loss(logits, targets, t)
    print(kd_loss)
    
    loss = SoftMeritBasedLoss()
    kd_loss = loss(logits, targets, t)
    print(kd_loss)