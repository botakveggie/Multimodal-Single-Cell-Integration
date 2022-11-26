import scanpy as sc
import pandas as pd
import argparse

import h5py
import hdf5plugin

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

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
    ## RNA (inputs)
    print("reading datasets")
    rna = sc.read_text("gse/rna.tsv.gz")
    rna_meta = pd.read_csv("gse/rna_meta.tsv", sep="\t", index_col=0)
    rna.var = rna_meta

    rna_data = rna.to_df()
    rna_data = pd.DataFrame.transpose(rna_data)
    
    ### storing full set 
    print("saving rna data")
    rna_data.to_csv('gse/test_inputs_full.csv')

    ### obtaining same genes used
    gene = list(rna.obs_names) # gene ids in gse
    valid = []
    for i in gene_id:
        if i in gene:
            j = gene.index(i)
            valid.append(j)
    
    rna_data = rna_data.iloc[:,valid]
    rna_data.to_csv('gse/test_inputs_sub.csv')
    
    ## ADT (target)
    print("adt")
    base = importr('data.table')
    with localconverter(ro.default_converter + pandas2ri.converter):
        adt_data = base.fread("gse/adt.tsv.gz")
    adt_data = pd.DataFrame.transpose(adt_data)
    protein = [t.split(' (')[0] for t in adt_data.iloc[0,] ] # removing the brackets
    protein[15] = "CD105"  # ENG
    protein[12] = "HLA-DR" # HLA-DR-DP-DQ
    adt_data.columns = protein

    valid = []
    for i in protein_id:
        if i in protein:
            j = protein.index(i)
            valid.append(j)

    adt_data = adt_data.iloc[1:,valid]
    print ("saving adt data")
    adt_data.to_csv('gse/test_targets.csv')
    print("done")

def main():
    convert()
    # using the full one
    full_rna = pd.read_csv('gse/test_inputs_full.csv', index_col=0)
    full_rna = variance(full_rna)
    full_rna = red_pca(full_rna)
    full_rna_200301 = red_umap(full_rna, 200, 30, 1)
    full_rna_240501 = red_umap(full_rna, 240, 50, 1)
    
    full_rna_200301.to_csv("gse/test_full_inputs_200301.csv")
    full_rna_240501.to_csv("gse/test_full_inputs_240501.csv")

    # using reduced data
    rna = pd.read_csv('gse/test_inputs_sub.csv', index_col=0)
    rna = variance(rna)
    rna = red_pca(rna)
    rna_200301 = red_umap(rna, 200, 30, 1)
    rna_240501 = red_umap(rna, 240, 50, 1)
    
    rna_200301.to_csv("gse/test_red_inputs_200301.csv")
    rna_240501.to_csv("gse/test_red_inputs_240501.csv")


if __name__=="__main__":
    main()
