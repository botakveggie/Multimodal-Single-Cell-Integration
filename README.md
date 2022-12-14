# Multimodal Single Cell Integration
**ZB4171 Project (2022)**

Using CITE-seq data, this project aims to use encoder-decoder neural network to model the complex assocation between transcriptomics and proteomics and make meaningful predictions of surface protein levels with gene expression levels.

The method consists of 2 main components: 
1. Feature preprocessing using low variance filtering, PCA and UMAP - `preprocess/`
2. Encoder-decoder NN using PyTorch - `main/`

*Disclosure: The project is adapted from Kaggle Competition: Open Problems - Multimodal Single-Cell Integration (Predict how DNA, RNA & protein measurements co-vary in single cells) Part of the codes are modelled after publicly accessible notebooks on kaggle*

## Downloading data used
Data used for the training of model can be accessed on https://www.kaggle.com/competitions/open-problems-multimodal/data
```
cd data
kaggle competitions download -c open-problems-multimodal # requiries Kaggle API
```

## Example of using the code
```
cd Multimodal-Single-Cell-Integration

# preprocessing of data
python3 preprocess/aws_preprocess.py

# training data; set VAL_FRAC = 0 in main/parameters.py for full model
python3 main/train.py --inputs_path "data/train_cite_inputs_reduced.csv" --targets_path "data/train_cite_targets.csv" --model "model/model.pth"

# getting prediction
python3 main/get_predictions.py --inputs_path "data/train_cite_inputs_reduced.csv" --model "model/model.pth" --outputs_path "outputs/outputs.csv" 
```

## Folder organisation
```
Multimodal-Single-Cell-Integration
│   README.md
│   metadata_explore.ipynb  # exploring data
|   script.ipynb            # exploring feature reductions
|
└───preprocess
|   |    aws_preprocess.py   # preprocessing for full data
|   |    preprocess.py       # preprocessing for subset of data
|   |    preprocess_umap.py      
|   |    preprocess_utils.py     
|  
└───main  
|   |   train.py            # does hold-out train/test split
|   |   cv.py               # cross validation
|   |   parameters.py
|   |   models.py
|   |   utils.py
|   |   get_predictions.py 
|   |   notebook.ipynb      # exploring NN model
│
└───metadata
│   │   metadata_cite_day_2_donor_27678.csv
│   │   metadata.csv
│
└───data #not on github
│   │   train_cite_targets.h5
│   │   train_cite_inputs.h5
│   │   train_cite_targets.csv
│   │   train_cite_inputs_reduced.csv
|
└───model
│   │   model_200501_final.pth     # final model used
```


