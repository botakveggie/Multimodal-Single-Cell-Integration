import numpy as np
import pandas as pd
import tables
import hdf5plugin

def main():
    #loading dataset
    dataset = pd.read_hdf('data/train_cite_targets.h5') 
    dataset.to_csv('data/train_cite_targets.csv')

if __name__=="__main__":
    main()
