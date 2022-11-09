import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class CiteDataset(Dataset):
    """
    A pytorch dataset class that accepts an inputs path, and optionally a targets path. 
    This class holds all the data and implements a __getitem__ method to be used by a 
    Python generator object or other classes that need it.

    Properties:
        pd_csv: bool, default=False
            Whether the input data is a csv file
        inputs: h5py.File or pandas.DataFrame
            Either a h5py.File or pd.DataFrame object containing the inputs.
        targets: h5py.File or pandas.DataFrame
            Either a h5py.File or pd.DataFrame object containing the targets.
        num_cells: int
            Number of rows in the Dataset.
        num_features: int or None
            Number of features in the input data.
        num_cells: int or None
            Number of columns in the target.
    """
    def __init__(self, inputs_path: str, targets_path: str or None = None):
        """
        Read the content of the inputs file
        Args:
            inputs_path: string
                Path to the inputs file.
            targets_path: string, default=None
                Path to the targets file.
        """
        file_extension = inputs_path.strip().split('.')[-1]
        self.pd_csv = (file_extension =="csv")
        if self.pd_csv: 
            # Initialises CiteDataset from pandas csv

            # Store the values into self.inputs
            self.inputs = pd.read_csv(inputs_path, index_col=0)

            # Values that inform the shape of the input matrix
            self.num_cells = self.inputs.shape[0]
            self.num_features = self.inputs.shape[1]
            if targets_path: # Init CiteDataset for training
                self.targets = pd.read_csv(targets_path, index_col=0)

                # Values that inform the shape of the targets matrix
                self.num_targets = self.targets.shape[1]
        self.protein_ids = list(self.targets.columns)
        self.cell_ids = list(self.targets.index)

    def data_size(self):
        """
        A function to inform the dimensions of the data. The function returns 
        a tuple of two integers:
            num_features: int
                Number of features in the input data.
            num_targets: int
                Number of target outputs.
        """
        return self.num_features, self.num_targets
    
    def __len__(self):
        """
        Return the number of instances in the data
        """
        return self.num_cells

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (inputs, targets), where inputs and targets are PyTorch tensors.
        """
        input_row = self.inputs.iloc[i,:]
        inputs = torch.tensor(input_row) 
        targets = None
        if hasattr(self, 'targets'):
            targets = torch.tensor(self.targets.iloc[i,:])
        
        return inputs, targets

class FCBlock(nn.Module):
    "A single feed-forward model with num_features input nodes and  num_hidden output nodes, with a specified dropout rate."
    def __init__(self, num_features: int, num_hidden: int, dropout: float):
        super(FCBlock, self).__init__()
        self.Input_Layer = nn.Linear(num_features, num_hidden)
        self.Dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        "Performs Components in sequence"
        x = self.Input_Layer(x)
        x = F.relu(x)
        x = self.Dropout(x)
        return x

class Encoder(nn.Module):
    """
    Encoder module to generate embeddings of a RNA vector
    """
    def __init__(self, num_features: int, dropout: float):
        super().__init__()
        self.l0 = FCBlock(num_features, 120, dropout)
        self.l1 = FCBlock(120, 60, dropout)
        self.l2 = FCBlock(60, 30, dropout)
        
    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        return x

class Decoder(nn.Module):
    """
    Decoder module to extract Protein sequences from RNA embeddings
    """
    def __init__(self, num_targets: int, dropout: float):
        super().__init__()
        self.l0 = FCBlock(30, 70, dropout)
        self.l1 = FCBlock(70, 100, dropout)
        self.l2 = FCBlock(100, num_targets, dropout)
        
    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        return x
    
class CiteseqModel(nn.Module):
    """
    Wrapper for the Encoder and Decoder modules
    Converts RNA sequence to Protein sequence
    """
    def __init__(self, num_features: int, num_targets: int, dropout: float):
        super().__init__()
        self.encoder = Encoder(num_features)
        self.decoder = Decoder(num_targets)
        
    def forward(self, x):
        embeddings = self.encoder(x)
        outputs = self.decoder(embeddings)
        return outputs