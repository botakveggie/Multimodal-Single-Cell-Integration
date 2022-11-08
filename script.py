import numpy as np
import pandas as pd
import h5py
import hdf5plugin

import umap 
import sklearn
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# loading data
df_metadata = pd.read_csv("metadata/metadata.csv")
df_metadata = df_metadata.loc[df_metadata.technology=="citeseq"]

## training subset: obtaining subset of size n per day per sample
x = list([13176,31800,32606]) # note:  donors 13176, 31800, and 32606 (train), 27678 (test)
y = list([2,3,4]) # day 2,3,4 in train set, day 7 test set
df_metadata_subset = pd.DataFrame(columns=df_metadata.columns)
for donor in x:
    for day in y:
        df = df_metadata.loc[(df_metadata.donor == donor) & (df_metadata.day == day)]
        df = df.iloc[:1500, :] # change this to change size 
        df_metadata_subset = pd.concat([df_metadata_subset, df], axis=0, ignore_index=True)

## train cite inputs
f = h5py.File("data/train_cite_inputs.h5",'r')  
gene_id = f['train_cite_inputs']['axis0'][:]
gene_id = [gene_id[i].decode('UTF-8') for i in range(22050)] # converts gene_id from bytes to str
cell_id = f['train_cite_inputs']['axis1'][:]
cell_id = [cell_id[i].decode('UTF-8') for i in range(70988)] # converts cell_id from bytes to str

## Find the row indexes for the cell_ids in the selected subset, from the h5 file. 
row_indexes = [cell_id.index(id) for id in df_metadata_subset['cell_id']] 
row_indexes.sort() # array can only be accessed with ordered index
x = f['train_cite_inputs']['block0_values'][row_indexes] # retrieves all the input rows for the selected cell_ids

new_cell_order = [cell_id[i] for i in row_indexes] # retrieving cell_ids of the input rows in sorted order
df_cite_input = pd.DataFrame(x, columns=gene_id, index=new_cell_order) 

## updating metadata with the same cell id order as the subset df
df_metadata_subset = df_metadata_subset.set_index('cell_id')
df_metadata_subset = df_metadata_subset.reindex(new_cell_order) 

f.close()
## train cite targets
f = h5py.File("data/train_cite_targets.h5",'r')
protein_id = f['train_cite_targets']['axis0'][:]
protein_id = [protein_id[i].decode('UTF-8') for i in range(140)]

x = f['train_cite_targets']['block0_values'][row_indexes] # same index/ cell order is used to retrieve corr rows
df_cite_target = pd.DataFrame(x, columns=protein_id, index=new_cell_order)

f.close()


# Low variance filter - dropping columns with <0.5 variance
variance = df_cite_input.var() # computes variance
columns = df_cite_input.columns
variable = [ ]
for i in range(0,len(variance)):
        if variance[i]>= 0.5: #setting the threshold as 0.5
            variable.append(columns[i])
reduced_data = df_cite_input[variable] 

# PCA
new_pca = PCA(n_components=None)
pca_data = new_pca.fit_transform(reduced_data)
## obtaining PCA with sum 0.9 variance
ACC_VAR = 0
for i, var in enumerate(new_pca.explained_variance_ratio_):
    ACC_VAR+=var
    # print(var)
    if ACC_VAR > 0.9 : 
        break
data_for_umap = pd.DataFrame(pca_data[:, 0:(i+1)], index=reduced_data.index) 

# UMAP - def components=2, neigh=15, min_dist=0.1
manifold = umap.UMAP(random_state = 4171,
                        n_components=240,
                        n_neighbors=50,
                        min_dist=0.3).fit(data_for_umap)
umap_data = manifold.transform(data_for_umap) # reduced data for NN

