import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

def adjust_learning_rate(optimizer, epoch):
    return

class loss_uniform_spread_l2(nn.Module):
    def __init__(self, lambd_centroid=0.04,
                           lambd_close=0.01,
                           weight_factor=10.0,
                           weight_factor_upper=10.0,
                           weight_factor_lower=10.0,
                           *args, **kwargs) -> None:
        super(loss_uniform_spread_l2, self).__init__(*args, **kwargs)
        self.lambd_centroid = lambd_centroid
        self.lambd_close = lambd_close
        self.weight_factor = weight_factor
        self.weight_factor_upper = weight_factor_upper
        self.weight_factor_lower = weight_factor_lower
        
    def forward(self,y_pred,y_true):
        n = y_true.size(0)
        mse = torch.mean((y_true - y_pred)**2)
        penalty_close = self.lambd_close * torch.sum(torch.abs(y_pred[:-1] - 
                                                               y_pred[1:]))
        targets_mean = torch.mean(y_true)
        penalty_centroid = self.lambd_centroid * torch.mean(torch.abs(y_pred - 
                                                                      targets_mean))
        # Give reward for higher weight to vert fast rr and lower rr
        conditions = [(y_true > 20), (y_true<18)]
        weight = torch.where(torch.logical_or(*conditions), self.weight_factor, 1.0) 
        weighted_mse_loss = torch.mean(weight * (y_pred - y_true)**2)
        
        weight_upper = torch.where(y_true >= 20, self.weight_factor_upper, 1.0) 
        weighted_upper_mse_loss = torch.mean(weight_upper * (y_pred - y_true)**2)
        weight_lower = torch.where(y_true <18 , self.weight_factor_lower, 1.0) 
        weighted_lower_mse_loss = torch.mean(weight_lower * (y_pred - y_true)**2)
        
        
        modified_loss = mse + penalty_centroid +\
                        penalty_close + \
                        weighted_lower_mse_loss +\
                        weighted_upper_mse_loss
        return modified_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)


class ClosenessLoss(nn.Module):
    def __init__(self, prob_factor=0.8, dist_factor=0.8,
                 weight_factor_upper = 5.0,
                 weight_factor_lower = 5.0,
                 class_weights=None):
        super(ClosenessLoss, self).__init__()
        self.prob_factor = prob_factor
        self.dist_factor = dist_factor
        self.weight_factor_upper = weight_factor_upper
        self.weight_factor_lower = weight_factor_lower
        self.class_weights = class_weights
        
    def forward(self, outputs, targets):
        # Compute the standard cross-entropy loss
        ce_loss = F.cross_entropy(outputs, targets, weight=self.class_weights)       
        
        # Encourages the predicted probabilities to be closer to the target probabilities.
        prob_output = F.softmax(outputs, dim=1)
        
        # Compute the penalty for being farther from the target
        dist = torch.dist(prob_output,targets)
        dist_loss = dist
        
        
        # weight_upper = torch.where(targets > 21, self.weight_factor_upper, 0.0) 
        # weighted_upper_mse_loss = torch.mean(weight_upper * (prob_output - targets)**2)
        # weight_lower = torch.where(targets <18 , self.weight_factor_lower, 0.0) 
        # weighted_lower_mse_loss = torch.mean(weight_lower * (prob_output - targets)**2)
        # Compute the frequencies of targets in the desired range
        target_freq = (targets >= 18) & (targets <= 21)
        target_freq = target_freq.float().sum() / len(targets)
        
        # Update the penalty weights based on target frequency
        self.weight_factor_lower *= (1 - target_freq)
        self.weight_factor_upper *= (1 - target_freq)
        
        weight_upper = torch.where(targets > 21, self.weight_factor_upper, 0.0) 
        weighted_upper_mse_loss = torch.mean(weight_upper * (prob_output - targets)**2)
        weight_lower = torch.where(targets <18 , self.weight_factor_lower, 0.0) 
        weighted_lower_mse_loss = torch.mean(weight_lower * (prob_output - targets)**2)
        
        
        
        # Reshape prob_output to have the same shape as prob_target by inserting a new dimension at position 1
        prob_output = prob_output.unsqueeze(1) 
        # prob_target = F.one_hot(torch.tensor(targets, dtype=torch.long), 
        #                         num_classes=outputs.size(1)).float()
        prob_target = F.one_hot(targets.clone().detach().long(),
                                num_classes=outputs.size(1)).float()
        diff = torch.abs(prob_output - prob_target).sum(dim=1).mean()
        penalty = self.prob_factor * (diff) +\
                    self.dist_factor * dist_loss -\
                    weighted_lower_mse_loss * self.weight_factor_lower -\
                    weighted_upper_mse_loss * self.weight_factor_upper
        
        # Combine the cross-entropy loss and penalty
        return ce_loss + penalty