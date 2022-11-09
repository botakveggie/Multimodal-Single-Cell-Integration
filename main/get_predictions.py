# python3 main/get_predictions.py --inputs_path data/train_cite_inputs_reduced.csv  --model_path ./pre-trained-models/_model.pth --output_path ./outputs/dev.txt
import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader

from models import CiteDataset, CiteseqModel
from utils import collator
from parameters import VERBOSE, device_str, DROPOUT

torch.manual_seed(0)

def get_predictions(model, dataset: CiteDataset, device='cpu'):
    """
    Function to test model on the test dataset. 
    """
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    preds_list = []
    with torch.no_grad():
        for _,data in enumerate(data_loader):
            inputs = data[0].to(device)
            # print((texts.shape))
            outputs = model(inputs).to(device)
            preds_list.append(outputs)
            # get the label predictions'
    preds_tensor = torch.cat(preds_list, dim=0)
    preds = pd.DataFrame(preds_tensor, columns=dataset.protein_ids, index=dataset.cell_ids)
    return preds

def get_test_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs_path', help='path to the inputs file')
    parser.add_argument('--model_path', required=True, help='path to the model file')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    parser.add_argument('--device_str', default='cpu', help='option for gpu acceleration. M1 Macbooks can use `mps`')
    return parser.parse_args()

def main(args):
    # if args.hasattr('device_str'): 
    #     device_str = args.device_str # uses gpu if device is specified
    #     print('Using gpu:', device_str)
    assert args.inputs_path is not None, "Please provide the inputs file using the --inputs_path argument"
    assert args.model_path is not None, "Please provide the model to test using --model_path argument"
    checkpoint = torch.load(args.model_path)    

    dataset = CiteDataset(args.inputs_path)
    num_features = dataset.data_size()[0]
    num_targets = len(checkpoint['protein_ids'])
    dataset.__setattr__('protein_ids', checkpoint['protein_ids'])
    dataset.__setattr__('num_targets', num_targets)
    if VERBOSE:
        print('Testing model with {} features. Trained for {} epochs'.format(num_features, checkpoint['epoch']))
    model = CiteseqModel(num_features, num_targets, DROPOUT)
    model.load_state_dict(checkpoint['model_state_dict'])

    preds = get_predictions(model, dataset, device_str)
    preds.to_csv(args.output_path)

if __name__=="__main__":
    args = get_test_arguments()
    main(args)

