import argparse
import pandas as pd
import numpy as np

def get_test_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_path', help='path to the inputs file')
    parser.add_argument('--outputs_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

def main(args):
    assert args.preds_path is not None, "Please provide the inputs file using the --preds_path argument"
    assert args.outputs_path is not None, "Please provide the model to test using --outputs_path argument"
    
    true = pd.read_csv(args.outputs_path)
    preds = pd.read_csv(args.preds_path)
    preds = preds.loc[:,true.columns]
    
    true.rename(columns={'Unnamed: 0':'cell_id'}, inplace=True) # renaming to first column to 'cell_id'
    true = true.melt(id_vars=['cell_id'], value_name='target', var_name='gene_id')
    
    preds.rename(columns={'Unnamed: 0':'cell_id'}, inplace=True) # renaming to first column to 'cell_id'
    preds = preds.melt(id_vars=['cell_id'], value_name='target', var_name='gene_id')

    score = np.corrcoef(true['target'].astype(float), preds['target'].astype(float))[1,0]
    
    print('Correlation: {:.5f}'.format(score))

if __name__=="__main__":
    args = get_test_arguments()
    main(args)
