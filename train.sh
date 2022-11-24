echo "training model with different parameters"

echo "running with umap 200 30 1"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap200301.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap200301.pth'

echo "running with umap 240 30 1"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap240301.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap240301.pth'

echo "running with umap 240 50 1"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap240501.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap240501.pth'

