# set your parameters here
VERBOSE = 1
LEARNING_RATE = 0.0005
BATCH_SIZE = 50
NUM_EPOCHS = 25
DROPOUT = 0.05
VAL_FRAC = 0.0  # 0 to train with all data
device_str = 'cpu'

LOSS = "CorrError"  # insert "CorrError" for correlation loss function

## do cross validation; set CV to 1; 0 to train with all data
CV = 0
FOLD = 5
