import numpy as np
import pandas as pd

import umap 
import sklearn
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

AWS = False

if not AWS:
    import h5py
    import hdf5plugin

def main():
    # loading data
    ## df_test_input = pd.read_csv("data/test_cite_inputs.csv", index_col=0)

    df_metadata = pd.read_csv("metadata/metadata.csv")
    df_metadata = df_metadata.loc[df_metadata.technology=="citeseq"]

    ## test subset: obtaining subset of size n per day 
    x = list([27678]) # note:  donors 13176, 31800, and 32606 (train), 27678 (test)
    y = list([2,3,4]) # day 2,3,4 in train & test set
    df_metadata_subset = pd.DataFrame(columns=df_metadata.columns)
    for donor in x:
        for day in y:
            df = df_metadata.loc[(df_metadata.donor == donor) & (df_metadata.day == day)]
            df = df.sample(1500, random_state=4171, axis=0) # random state can be changed
            df_metadata_subset = pd.concat([df_metadata_subset, df], axis=0, ignore_index=True)

    ## train cite inputs
    if not AWS:
        f = h5py.File("data/test_cite_inputs.h5",'r') 
        gene_id = f['test_cite_inputs']['axis0'][:]
        gene_id = [gene_id[i].decode('UTF-8') for i in range(22050)] # converts gene_id from bytes to str
        cell_id = f['test_cite_inputs']['axis1'][:]
        cell_id = [cell_id[i].decode('UTF-8') for i in range(48663)] # converts cell_id from bytes to str
    else:    
        f = pd.read_hdf("data/test_cite_inputs.h5") 
        gene_id = list(f.columns)
        cell_id = list(f.index)
        
    ## Find the row indexes for the cell_ids in the selected subset, from the h5 file. 
    row_indexes = [cell_id.index(id) for id in df_metadata_subset['cell_id']] 
    row_indexes.sort() # array can only be accessed with ordered index

    new_cell_order = [cell_id[i] for i in row_indexes] # retrieving cell_ids of the input rows in sorted order
    
    if not AWS:
        x = f['test_cite_inputs']['block0_values'][row_indexes] # retrieves all the input rows for the selected cell_ids
        df_test_input = pd.DataFrame(x, columns=gene_id, index=new_cell_order) 
    else:
        df_test_input = f.loc[row_indexes]


    # Low variance filter - dropping columns with <0.5 variance
    variance = df_test_input.var() # computes variance
    columns = df_test_input.columns
    variable = [ ]
    for i in range(0,len(variance)):
            if variance[i]>= 0.5: #setting the threshold as 0.5
                variable.append(columns[i])
    reduced_data = df_test_input[variable] 

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

    # saving umap data to csv
    pd.DataFrame(umap_data, index=reduced_data.index).to_csv("data/test_cite_inputs_reduced.csv")
if __name__ == "__main__":
    main()
