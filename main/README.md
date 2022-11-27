# Encoder-Decoder Neural Network Model (PyTorch)

takes in preprocessed input/RNA data as input

**Components**
1. `models.py` contains classes used by the model
    - CiteDataset; FCBlock; Encoder; RelativeEncoder; Decoder; CiteseqModel
2. `utils.py` contains the loss functions and collator function used by the model

3. `train.py` and `cv.py` trains the model and validates the model with hold-out validation and cross-validation respectively.
    - `parameters.py` contains considered for optimisation

4. `get_predictions.py` takes in the trained model path and preprocessed inputs and returns the predictions.
