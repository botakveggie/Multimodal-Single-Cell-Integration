# python3 main/train.py --inputs_path data/train_cite_inputs_reduced.csv --targets_path data/train_cite_targets_reduced.csv --model_path ./pre-trained-models
import datetime
import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split

from models import CiteDataset, CiteseqModel
from utils import collator,correlation_score, CorrError
from parameters import VERBOSE, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, DROPOUT, device_str, VAL_FRAC, FOLD, LOSS, CV

torch.manual_seed(0)

def train(model, dataset, train_set, validation_set, fold, batch_size, learning_rate, num_epoch, device='cpu', model_path=None, loss_fn=nn.MSELoss, optim = optim.Adam):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.
    """
    data_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collator, shuffle=True)

    # assign these variables
    criterion=loss_fn()
    optimizer = optim(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    model.train()
    start = datetime.datetime.now()
    # best_epoch_loss = 1
    for epoch in range(num_epoch):
        loss = 0.0
        ## training one epoch
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
            loss += loss_tensor.item()
            
        scheduler.step()
        if VERBOSE == 1:
            print('[Epoch %d, Step %5d] Loss: %.5f' %
                (epoch + 1, step + 1, loss/(step + 1)))
    
    if CV: # Evaluates one epoch
        model.eval()
        data_loader_val = DataLoader(validation_set, batch_size=20, collate_fn=collator, shuffle=False)
        with torch.no_grad():
            val_loss = 0.0
            for step,data in enumerate(data_loader_val):
                inputs = data[0].to(device)
                truths = data[1].to(device)
                outputs = model(inputs).to(device)
                val_loss += criterion(outputs, truths).item()

        print('[validation loss] Loss: {:.5f}'.format(val_loss/(step + 1)))

        score=val_loss/(step + 1)

        # if (VERBOSE == 1) & (score <= best_epoch_loss):
        #     print("Validation Loss Improved ({:.5f} -> {:.5f})".format(best_epoch_loss, score))
        #     best_epoch_loss = score

    end = datetime.datetime.now()
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))
    
    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = {
        'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'protein_ids': dataset.protein_ids,
        'params': {'VERBOSE': VERBOSE, 'LEARNING_RATE': LEARNING_RATE, 'BATCH_SIZE': BATCH_SIZE, 'NUM_EPOCHS': NUM_EPOCHS, 'DROPOUT': DROPOUT}
    }
    
    model_path = os.path.join(model_path, f"model_f{fold+1}.pth")
    torch.save(checkpoint, model_path)
    print('Model saved in ', model_path)
    
    if CV:
        return score

def kfold_split(dataset, fold=FOLD):
    fold_sizes = [len(dataset) // fold] * (fold - 1) + [len(dataset) // fold + len(dataset) % fold]
    dataset_folds = torch.utils.data.random_split(dataset, fold_sizes, generator=torch.Generator().manual_seed(42))
    for fold in range(fold):
        yield torch.utils.data.ConcatDataset(dataset_folds[:fold] + dataset_folds[fold + 1:]), dataset_folds[fold]

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
    
    ## init dataset
    dataset = CiteDataset(args.inputs_path, args.targets_path)
    num_features, num_targets = dataset.data_size()
    if VERBOSE:
        print(f'Training model for {num_features} features for {NUM_EPOCHS} epochs')

    ## type of loss fn to use
    if LOSS == "CorrError":
        loss_fn = CorrError

    # cross validation
    if CV:
        scores = []
        for fold, (ds_train, ds_eval) in enumerate (kfold_split(dataset)):
            print("Fold: ", fold+1)
            ## init model
            model = CiteseqModel(num_features, num_targets, DROPOUT)
            score = train(model, dataset, ds_train, ds_eval, fold, BATCH_SIZE ,LEARNING_RATE, NUM_EPOCHS, device_str, args.model_path, loss_fn)
            scores.append(score)
        print('CV score:', -np.mean(scores))
    else: # train full model
        model = CiteseqModel(num_features, num_targets, DROPOUT)
        train(model, dataset, dataset, None , 'inal', BATCH_SIZE ,LEARNING_RATE, NUM_EPOCHS, device_str, args.model_path, loss_fn)

if __name__=="__main__":
    args = get_train_arguments()
    main(args)

