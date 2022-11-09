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

    batch_inputs = torch.cat([tens.unsqueeze(0) for tens in inputs])
    if targets: # for training, return both texts and batch_targets
        batch_targets = torch.cat([tens.unsqueeze(0) for tens in targets])
    return batch_inputs, batch_targets
