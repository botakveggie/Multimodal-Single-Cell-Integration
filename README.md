# Multimodal Single Cell Integration
 ZB4171 Project

Model is stored in main.

Folder organisation
```
Multimodal-Single-Cell-Integration
│   README.md
│   metadata_explore.ipynb  # exploring data
|   script.ipynb            # exploring feature reductions
|
|   aws_preprocess.py       # preprocessing for full data
|   preprocess.py           # preprocessing for subset of data
|   preprocess_umap.py      # only umap
|  
└───main  #
|   |   train.py            # does hold-out validation
|   |   cv.py               # cross validation
|   |   parameters.py
|   |   models.py
|   |   utils.py
|   |   get_predictions.py
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
│   │   model_final.pth
|
└──output
│   │   
|


```
