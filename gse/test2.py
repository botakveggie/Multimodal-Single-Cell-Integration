import scanpy as sc
import pandas as pd
import argparse

import h5py
import hdf5plugin

import anndata as ad

from preprocess_utils import variance, red_pca, red_umap

def convert():
    print("starting")
    # obtaining rna and protein rows used for the model
    f = h5py.File("data/train_cite_inputs.h5",'r') 
    gene_id = f['train_cite_inputs']['axis0'][:]
    gene_id = [gene_id[i].decode('UTF-8') for i in range(22050)]
    gene_id = [t.split('_')[1] for t in gene_id ]
    f.close()

    f = h5py.File("data/train_cite_targets.h5",'r')
    protein_id = f['train_cite_targets']['axis0'][:]
    protein_id = [protein_id[i].decode('UTF-8') for i in range(140)]
    f.close() 

    # reading in test set
    print("reading datasets")
    adata = ad.read_h5ad("gse/GSE194122_cite.h5ad")

    ## RNA (inputs)
    rna_names = adata.var[adata.var["feature_types"] == "GEX"].index
    rna = adata.X[:, :13953]
    rna = pd.DataFrame(rna.toarray(), index=adata.obs_names, columns=rna_names)
    rna = rna.sample(70000, random_state=4171)
    
    ### storing full set 
    print("saving rna data")
    rna.to_csv('gse/test_inputs_full2.csv')

    ### obtaining same genes used
    valid = []
    for i in gene_id:
        if i in rna_names:
            valid.append(i)
    
    rna = rna.loc[:,valid]
    rna.to_csv('gse/test_inputs_sub2.csv')

    ## ADT (target)
    print("adt")
    adt_names = adata.var[adata.var["feature_types"] == "ADT"].index
    adt = adata.X[:, 13953:]
    adt = pd.DataFrame(adt.toarray(), index=adata.obs_names, columns=adt_names)
    adt = adt.sample(70000, random_state=4171)

    valid = []
    for i in protein_id:
        if i in adt_names:
            valid.append(i)
    print(len(valid))
    adt = adt.loc[:,valid]
    print ("saving adt data")
    adt.to_csv('gse/test_targets2.csv')
    print("done")

def main():
    convert()
    # using the full one
    full_rna = pd.read_csv('gse/test_inputs_full2.csv', index_col=0)

    full_rna = variance(full_rna)
    full_rna = red_pca(full_rna)
    full_rna_200301 = red_umap(full_rna, 200, 30, 1)
    full_rna_240501 = red_umap(full_rna, 240, 50, 1)
    
    full_rna_200301.to_csv("gse/test_inputs_200301.csv")
    full_rna_240501.to_csv("gse/test_inputs_240501.csv")

    # using reduced data
    rna = pd.read_csv('gse/test_inputs_sub2.csv', index_col=0)
    rna = variance(rna)
    rna = red_pca(rna)
    rna_200301 = red_umap(rna, 200, 30, 1)
    rna_240501 = red_umap(rna, 240, 50, 1)
    
    rna_200301.to_csv("gse/test_inputs_200301r.csv")
    rna_240501.to_csv("gse/test_inputs_240501r.csv")


if __name__=="__main__":
    main()
