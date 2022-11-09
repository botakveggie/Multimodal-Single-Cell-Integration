import numpy as np
import pandas as pd

def main():
    # evaluation ids of citeseq-specific cells - (view script.ipynb to obtain)
    evaluation = pd.read_csv('submission/evaluation_ids_cite.csv')

    # test target values - cell_id is stored in first row
    target = pd.read_csv('output/base_preds.csv')

    # converting the format of output
    target.rename(columns={'Unnamed: 0':'cell_id'}, inplace=True) # renaming to first column to 'cell_id'
    target = target.melt(id_vars=['cell_id'], value_name='target', var_name='gene_id')

    # merging to find corr row id
    target = pd.merge(target, evaluation, how='left', on=['cell_id','gene_id'])
    target['row_id','target'].to_csv('output/target_submission.csv', index=False)

if __name__ == "__main__":
    main()

