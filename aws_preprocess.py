import numpy as np
import pandas as pd
import tables
import hdf5plugin

import umap 
import sklearn
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def main():
    #loading dataset
    print('loading metadata')
    dataset = pd.read_hdf('data/train_cite_inputs.h5') 
    print('metadata loaded')
    # Low variance filter - dropping columns with <0.5 variance
    print('loading train inputs')
    variance = dataset.var() # computes variance
    columns = dataset.columns
    variable = [ ]
    for i in range(0,len(variance)):
            if variance[i]>= 0.5: #setting the threshold as 0.5
                variable.append(columns[i])
    reduced_data = dataset[variable] 
    print('inputs loaded')

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
    print('Performing UMAP')

    # UMAP - def components=2, neigh=15, min_dist=0.1
    umap_data = umap.UMAP(random_state = 4171,
                            n_components=200,
                            n_neighbors=30,
                            min_dist=0.5).fit_transform(data_for_umap) # reduced data for NN

    res = pd.DataFrame(umap_data, index=reduced_data.index)
    res.to_csv('data/train_cite_inputs_reduced.csv')
    print('UMAP done. saving to: data/train_cite_inputs_reduced.csv')


if __name__=="__main__":
    main()