# Validation using existing CITE-Seq data

https://www.biorxiv.org/content/10.1101/2020.11.11.378620v1.full

Download the dataset from https://engraftable-hsc.cells.ucsc.edu/
```
wget https://cells.ucsc.edu/engraftable-hsc/mrna/exprMatrix.tsv.gz
wget https://cells.ucsc.edu/engraftable-hsc/mrna/meta.tsv

wget https://cells.ucsc.edu/engraftable-hsc/adt/exprMatrix.tsv.gz
wget https://cells.ucsc.edu/engraftable-hsc/adt/meta.tsv
```

1.  `test.py` preprocesses the downloaded RNA data 
    - `preprocess_utils.py` contains functions required to preprocess the input data
2.  `get_score.py` determines the correlation between predicted and true data 
    - prediction is made using `main/get_predictions.py`
