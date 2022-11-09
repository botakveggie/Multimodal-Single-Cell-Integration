# python3 aws_preprocess.py --inputs_path [inputs_path] --output_path [output_path]
import argparse
import numpy as np
import pandas as pd
import tables
import hdf5plugin

import umap 
import sklearn
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def feature_red(dataset):
    # Low variance filter - dropping columns with <0.5 variance
    variance = dataset.var() # computes variance
    columns = dataset.columns
    variable = [ ]
    for i in range(0,len(variance)):
            if variance[i]>= 0.5: #setting the threshold as 0.5
                variable.append(columns[i])
    reduced_data = dataset[variable] 

    # PCA
    new_pca = PCA(n_components=None)
    pca_data = new_pca.fit_transform(reduced_data)
    ## obtaining PCAs with sum 0.9 variance
    ACC_VAR = 0
    for i, var in enumerate(new_pca.explained_variance_ratio_):
        ACC_VAR+=var
        if ACC_VAR > 0.9 : 
            break
    data_for_umap = pd.DataFrame(pca_data[:, 0:(i+1)], index=reduced_data.index) 

    # UMAP - def components=2, neigh=15, min_dist=0.1
    umap_data = umap.UMAP(random_state = 4171,
                            n_components=200,
                            n_neighbors=30,
                            min_dist=0.5).fit_transform(data_for_umap) # reduced data for NN

    res = pd.DataFrame(umap_data, index=reduced_data.index)
    return res

def get_test_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs_path', help='path to the inputs file')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

def main(args):
    df_cite_input = pd.read_hdf(args.inputs_path) 
    res = feature_red(df_cite_input)
    
    # saving umap data to csv
    res.to_csv(args.output_path)

if __name__=="__main__":
    args = get_test_arguments()
    main(args)
    
