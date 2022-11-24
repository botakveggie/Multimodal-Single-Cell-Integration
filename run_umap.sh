# trying out different parameters for umap
echo "trying different numbers of components: 200, 300, 400, 500"

# python3 ./preprocess_umap.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --outputs_path 'data/train_cite_inputs_umap200501.csv' --components 200 --neighbours 50 --min_dist 1


# python3 ./preprocess_umap.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --outputs_path 'data/train_cite_inputs_umap300501.csv' --components 300 --neighbours 50 --min_dist 1


# python3 ./preprocess_umap.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --outputs_path 'data/train_cite_inputs_umap400501.csv' --components 400 --neighbours 50 --min_dist 1

# 500 disconnected
# python3 ./preprocess_umap.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --outputs_path 'data/train_cite_inputs_umap500501.csv' --components 500 --neighbours 50 --min_dist 1

echo "trying different numbers of neighbours: 30, 50, 70"

# python3 ./preprocess_umap.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --outputs_path 'data/train_cite_inputs_umap240701.csv' --components 240 --neighbours 70 --min_dist 1


# python3 ./preprocess_umap.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --outputs_path 'data/train_cite_inputs_umap300301.csv' --components 300 --neighbours 30 --min_dist 1


# python3 ./preprocess_umap.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --outputs_path 'data/train_cite_inputs_umap300701.csv' --components 300 --neighbours 70 --min_dist 1

echo "trying different min dist: 0.5 and 1.5"

python3 ./preprocess_umap.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --outputs_path 'data/train_cite_inputs_umap240300_5.csv' --components 240 --neighbours 30 --min_dist 0.5


python3 ./preprocess_umap.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --outputs_path 'data/train_cite_inputs_umap240301_5.csv' --components 240 --neighbours 30 --min_dist 1.5


python3 ./preprocess_umap.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --outputs_path 'data/train_cite_inputs_umap300300_5.csv' --components 300 --neighbours 30 --min_dist 0.5


python3 ./preprocess_umap.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --outputs_path 'data/train_cite_inputs_umap300301_5.csv' --components 300 --neighbours 30 --min_dist 1.5

