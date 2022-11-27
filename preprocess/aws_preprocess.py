import numpy as np
import pandas as pd
import datetime

import umap 
import sklearn
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def main():
    # converting targets.h5 to .csv
    dataset = pd.read_hdf('data/train_cite_targets.h5') 
    dataset.to_csv('data/train_cite_targets.csv')
    
    # loading dataset
    print('loading train inputs')
    dataset = pd.read_hdf('data/train_cite_inputs.h5') 
    print('inputs loaded')
    
    # Low variance filter - dropping columns with <0.5 variance
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
        if ACC_VAR > 0.9 : 
            break
    data_for_umap = pd.DataFrame(pca_data[:, 0:(i+1)], index=reduced_data.index) 
    
    ## saving pca data to csv
    pcaname = 'data/train_cite_inputs_PCA{}.csv'.format(i+1)
    print(f'saving to: {pcaname}')
    pd.DataFrame(data_for_umap, index=reduced_data.index).to_csv(pcaname)
    
    # UMAP 
    print('Performing UMAP')
    start = datetime.datetime.now()
    umap_data = umap.UMAP(random_state = 4171,
                            n_components=200,
                            n_neighbors=30,
                            min_dist=1,
                            n_epochs=200,
                            spread=2).fit_transform(data_for_umap) # reduced data for NN

    end = datetime.datetime.now()
    print('UMAP finished in {} minutes.'.format((end - start).seconds / 60.0))
    
    ## saving umap data to csv
    res = pd.DataFrame(umap_data, index=reduced_data.index)
    res.to_csv('data/train_cite_inputs_reduced.csv')
    print('UMAP done. saving to: data/train_cite_inputs_reduced.csv')

if __name__=="__main__":
    main()
