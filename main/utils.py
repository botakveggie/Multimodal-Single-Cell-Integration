import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

torch.manual_seed(0)

def mse_criterion(outputs, labels):
    """ MSE Loss function"""
    return nn.MSELoss()(outputs, labels)

def correlation_score(y_true, y_pred):
    """
    Scores the predictions according to the competition rules. 
    It is assumed that the predictions are not constant.
    Returns the average of each sample's Pearson correlation coefficient
    """
    
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)

def collator(batch):
    """
    Define a function that receives a list of (inputs, targets) pair for each cell
    and return a pair of tensors:
        batch_inputs: a tensor that combines all the inputs in the mini-batch
        batch_targets: a tensor that combines all the targets in the mini-batch
    """
    # batch_inputs tensor dimensions: batch_dim x INPUT_DIM
    # batch_targets tensor dimensions: batch_dim x num_targets
    inputs, targets = zip(*batch)
    batch_targets = []
    batch_inputs = torch.cat([tens.unsqueeze(0) for tens in inputs])
    if targets[0] is not None: # for training, return both texts and batch_targets
        batch_targets = torch.cat([tens.unsqueeze(0) for tens in targets])
    return batch_inputs, batch_targets

class CorrError():
    """ Pearson Correlation Loss Function """
    def __init__(self, reduction='mean', normalize=True):
        self.reduction, self.normalize = reduction, normalize

    def __call__(self, y, y_target):
        y = y - torch.mean(y, dim=1).unsqueeze(1)
        y_target = y_target - torch.mean(y_target, dim=1).unsqueeze(1)
        loss = -torch.sum(y * y_target, dim=1) / (y_target.shape[-1] - 1)  # minus because we want gradient ascend
        if self.normalize:
            s1 = torch.sqrt(torch.sum(y * y, dim=1) / (y.shape[-1] - 1))
            s2 = torch.sqrt(torch.sum(y_target * y_target, dim=1) / (y_target.shape[-1] - 1))
            loss = loss / s1 / s2
        if self.reduction == 'mean':
            return torch.mean(loss)
        return loss
