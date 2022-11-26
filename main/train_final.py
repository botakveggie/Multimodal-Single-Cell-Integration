import datetime
import argparse
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split

from models import CiteDataset, CiteseqModel
from utils import collator,correlation_score, CorrError

#using diff file
from parameter import VERBOSE, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, DROPOUT, device_str, VAL_FRAC, LOSS

torch.manual_seed(0)

def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None, loss_fn=nn.MSELoss, optim = optim.Adam):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    
    # assign these variables
    criterion=loss_fn()
    optimizer = optim(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    # train
    model.train()
    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        running_loss = 0.0
        for step, data in enumerate(data_loader):
            # get the inputs; data is a tuple of (inputs_tensor, targets_tensor)
            inputs = data[0].to(device)
            targets = data[1].to(device)

            model.zero_grad()
            # do forward propagation
            y_preds = model(inputs.float()).to(device)

            # do loss calculation
            loss_tensor = criterion(y_preds, targets)
            
            # do backward propagation
            loss_tensor.backward()
            # do parameter optimization step
            optimizer.step()

            # calculate running loss value
            running_loss += loss_tensor.item()
            
        scheduler.step()
        if VERBOSE == 1:
            print('[Epoch %d, Step %5d] Loss: %.5f' %
                (epoch + 1, step + 1, running_loss / (step+1)))
            running_loss = 0.0
    
    end = datetime.datetime.now()
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))
    
    # saving model
    checkpoint = {
        'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'protein_ids': dataset.protein_ids,
        'params': {'VERBOSE': VERBOSE, 'LEARNING_RATE': LEARNING_RATE, 'BATCH_SIZE': BATCH_SIZE, 'NUM_EPOCHS': NUM_EPOCHS, 'DROPOUT': DROPOUT}
    }
    
    torch.save(checkpoint,model_path)
    print('Model saved as', model_path)

def get_train_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs_path', help='path to the inputs file')
    parser.add_argument('--targets_path', default=None, help='path to the targets file')
    parser.add_argument('--model_path', required=True, help='path to save the model file after training')
    parser.add_argument('--device_str', default='cpu', help='option for gpu acceleration. M1 Macbooks can use `mps`')
    return parser.parse_args()

def main(args):
    if args.device_str is not None: 
        device_str = args.device_str # uses gpu if device is specified
        print('Using gpu:', device_str)
    if torch.cuda.is_available():
        device_str = 'cuda'
    assert args.inputs_path is not None, "Please provide the inputs file using the --inputs_path argument"
    assert args.targets_path is not None, "Please provide the targets file using the --targets_path argument"
    dataset = CiteDataset(args.inputs_path, args.targets_path)
    num_features, num_targets = dataset.data_size()
    if VERBOSE:
        print(f'Training model for {num_features} features for {NUM_EPOCHS} epochs')
    model = CiteseqModel(num_features, num_targets, DROPOUT)

    ## type of loss fn to use
    if LOSS == "CorrError":
        loss_fn = CorrError
        print("Using CorrError as loss fn")
    else: 
        loss_fn = nn.MSELoss
    train(model, dataset, BATCH_SIZE ,LEARNING_RATE, NUM_EPOCHS, device_str, args.model_path, loss_fn)

if __name__=="__main__":
    args = get_train_arguments()
    main(args)

