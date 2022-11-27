import numpy as np
import pandas as pd
import datetime

import umap 
import sklearn
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def variance(dataset):
    """Low variance filter - dropping columns with <0.5 variance
    Insert dataset with named index"""

    print('checking variance of columns in inputs')
    variance = dataset.var() # computes variance
    columns = dataset.columns
    variable = [ ]
    start = datetime.datetime.now()
    for i in range(0,len(variance)):
            if variance[i]>= 0.5: #setting the threshold as 0.5
                variable.append(columns[i])
    reduced_data = dataset[variable] 
    print('informative columns chosen')    
    end = datetime.datetime.now()

    print('variance filter finished in {} minutes.'.format((end - start).seconds / 60.0))
    return reduced_data

def red_pca(reduced_data):
    """ Performs PCA and obtains top PCs representing 0.9 variance """
     # PCA
    print('Performing PCA from {} features with considerable variance'.format(reduced_data.shape[1]))
    start = datetime.datetime.now()
    new_pca = PCA(n_components=None)
    pca_data = new_pca.fit_transform(reduced_data)
    end = datetime.datetime.now()
    print('PCA finished in {} minutes.'.format((end - start).seconds / 60.0))

    ## obtaining PCAs with sum 0.9 variance
    ACC_VAR = 0
    for i, var in enumerate(new_pca.explained_variance_ratio_):
        ACC_VAR+=var
        if ACC_VAR > 1 : 
            break
    data_for_umap = pd.DataFrame(pca_data[:, 0:(i+1)], index=reduced_data.index) 
    
    print (f'{i+1} Principal components chosen')
    return data_for_umap

def red_umap(reduced, components, neighbours, min_dist):
    """ Performs UMAP on PCA components """
    print('Performing UMAP')
    start = datetime.datetime.now()
    umap_data = umap.UMAP(random_state = 4171,
                            n_components=components,
                            n_neighbors=neighbours,
                            min_dist=min_dist,
                            n_epochs=200,
                            spread=2).fit_transform(reduced) # reduced data for NN

    # saving umap data to csv
    reduced_umap = pd.DataFrame(umap_data, index=reduced.index)
    print('UMAP done.')
    
    end = datetime.datetime.now()
    print('UMAP finished in {} minutes.'.format((end - start).seconds / 60.0))
    return reduced_umap

