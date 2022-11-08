# python3 main/train.py --inputs_path [inputs_path] --targets_path [targets_path] --model_path [model_path] --output_path [output_path]
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import CiteDataset, CiteseqModel
from utils import collator, get_train_arguments
from parameters import VERBOSE, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, DROPOUT, device_str

torch.manual_seed(0)

def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None, loss_fn=nn.MSELoss, optim = optim.Adam):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    # assign these variables
    criterion=loss_fn
    optimizer = optim(model.parameters(), lr=learning_rate)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    # train, validation split?
    # do here
    
    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0

        for step, data in enumerate(data_loader):
            # get the inputs; data is a tuple of (inputs_tensor, targets_tensor)
            inputs = data[0].to(device)
            targets = data[1].to(device)

            model.zero_grad()
            # do forward propagation
            y_preds = model(inputs)

            # do loss calculation
            loss_tensor = criterion(input= y_preds, target= targets)
            if VERBOSE == 1: print('Loss tensor:\n',loss_tensor)

            # do backward propagation
            loss_tensor.backward()
            # do parameter optimization step
            optimizer.step()

            # calculate running loss value
            running_loss += loss_tensor.item()
            # print('running loss updated to', running_loss)

            # print loss value every 100 steps and reset the running loss
                # input()
        # scheduler.step()
        if VERBOSE == 1:
            print('[Epoch %d, Step %5d] MSE loss: %.3f' %
                (epoch + 1, step + 1, running_loss / 100))
            running_loss = 0.0
    
        # Check correlation score with validation set
        # correlation_score()
    end = datetime.datetime.now()
    
    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = {
        'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        # 'loss':running_loss
    }
    torch.save(checkpoint, model_path)
    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))

def main(args):
    if args.device_str is not None: 
        device_str = args.device_str # uses gpu if device is specified
        print('Using gpu:', device_str)
    assert args.inputs_path is not None, "Please provide the inputs file using the --inputs_path argument"
    assert args.targets_path is not None, "Please provide the targets file using the --targets_path argument"
    dataset = CiteDataset(args.inputs_path, args.targets_path)
    num_features, num_targets = dataset.data_size()
    if VERBOSE:
        print(f'Training model for {num_features} features for {NUM_EPOCHS} epochs')
    model = CiteseqModel(num_features, num_targets, DROPOUT)
    train(model, dataset, BATCH_SIZE ,LEARNING_RATE, NUM_EPOCHS, device_str, args.model_path)

if __name__=="__main__":
    args = get_train_arguments()
    main(args)

