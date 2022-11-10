import argparse
import numpy as np
import pandas as pd

import umap 

def red_umap( reduced, components, neighbours, min_dist):
    print('Performing UMAP')
    umap_data = umap.UMAP(random_state = 4171,
                            n_components=components,
                            n_neighbors=neighbours,
                            min_dist=min_dist).fit_transform(reduced) # reduced data for NN

    # saving umap data to csv
    reduced_umap = pd.DataFrame(umap_data, index=reduced.index)
    print('UMAP done.')
    return reduced_umap

def get_train_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs_path', help='path to the inputs file')
    parser.add_argument('--outputs_path', default=None, help='path to the output file')
    parser.add_argument('--components', default=200, help='insert number of components')
    parser.add_argument('--neighbours', default=50, help='insert number of neighbours')
    parser.add_argument('--min_dist', default=0.5, help='insert number of min distance')
    return parser.parse_args()


def main(args):
    assert args.inputs_path is not None, "Please provide the inputs file using the --inputs_path argument"
    assert args.outputs_path is not None, "Please provide the outputs file using --outputs_path argument"
    
    print("reading file")
    RED_PCA = pd.read_csv(args.inputs_path, index_col=0)

    print("running UMAP")
    res = red_umap(RED_PCA, args.components, args.neighbours, args.min_dist)

    print("saving file")
    res.to_csv(args.outputs_path)

    
if __name__=="__main__":
    args = get_train_arguments()
    main(args)
