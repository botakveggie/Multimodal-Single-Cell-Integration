import numpy as np
import pandas as pd

def main():
    # evaluation ids of citeseq-specific cells - (view script.ipynb to obtain)
    evaluation = pd.read_csv('submission/evaluation_ids_cite.csv')

    # test target values
    target = pd.read_csv('output/base_preds.csv', index_col=0)

    # converting the format of output
    target['cell_id'] = target.index # adding cell_id as a column
    target = target.melt(id_vars=['cell_id'], value_name='target', var_name='gene_id')

    # merging to find corr row id
    target = pd.merge(target, evaluation, how='left', on=['cell_id','gene_id'])
    target = target['row_id','target']
    target.to_csv('output/target_submission.csv', index=False)

if __name__ == "__main__":
    main()