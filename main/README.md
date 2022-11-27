# Encoder-Decoder Neural Network Model (PyTorch)

takes in preprocessed input/RNA data as input

**Components**
1. `models.py` contains classes used by the model
    - CiteDataset; FCBlock; Encoder; RelativeEncoder; Decoder; CiteseqModel
2. `utils.py` contains the loss functions and collator function used by the model
3. `train.py` splits dataset into train and test set 
4. `cv.py` trains the model with train set and validates the model with cross-validation 
5. `parameters.py` to specify parameters used in the model
6. `get_predictions.py` takes in the trained model path and preprocessed inputs and returns the predictions.
