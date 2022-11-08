# python3 main/get_predictions.py --inputs_path [inputs_path] --model_path [model_path] --output_path [output_path]
import pandas as pd

import torch
from torch.utils.data import DataLoader

from models import CiteDataset, CiteseqModel
from utils import mse_criterion, collator, correlation_score, get_test_arguments
from parameters import VERBOSE, device_str

torch.manual_seed(0)

def get_predictions(model, dataset: CiteDataset, device='cpu'):
    """
    Function to test model on the test dataset. 
    """
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    preds_tensor = None
    with torch.no_grad():
        for _,data in enumerate(data_loader):
            targets = data[0].to(device)
            # print((texts.shape))
            outputs = model(targets).cpu()
            if preds_tensor is None:
                preds_tensor = outputs
            else:
                torch.cat((preds_tensor, outputs), dim=0)
            # get the label predictions
    preds = pd.DataFrame(preds_tensor, columns=dataset.protein_ids, index=dataset.cell_ids)
    return preds

def main(args):
    if args.device_str is not None: 
        device_str = args.device_str # uses gpu if device is specified
        print('Using gpu:', device_str)
    assert args.inputs_path is not None, "Please provide the inputs file using the --inputs_path argument"
    assert args.model_path is not None, "Please provide the model to test using --model_path argument"
    checkpoint = torch.load(args.model_path)    

    dataset = CiteDataset(args.inputs_path)
    num_features, num_targets = dataset.data_size()
    if VERBOSE:
        print('Testing model with {} features. Trained for {} epochs'.format(num_features, checkpoint['epoch']))
    model = CiteseqModel(num_features, num_targets)
    preds = get_predictions(model, dataset, device_str)
    preds.to_csv(args.output_path)

if __name__=="__main__":
    args = get_test_arguments()
    main(args)

