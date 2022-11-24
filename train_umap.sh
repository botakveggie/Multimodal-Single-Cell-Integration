echo "training model with different parameters"

echo "by components"
echo "running with umap 200 50 1"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap200501.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap200501.pth'

echo "running with umap 240 50 1"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap240501.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap240501.pth'

echo "running with umap 300 50 1"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap300501.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap300501.pth'

echo "running with umap 400 50 1"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap400501.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap400501.pth'


echo "by neighbours"
echo "running with umap 240 30 1"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap240301.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap240301.pth'

echo "running with umap 240 70 1"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap240701.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap240701.pth'

echo "running with umap 300 30 1"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap300301.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap300301.pth'

echo "running with umap 300 70 1"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap300701.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap300701.pth'

echo "by distance"
echo "running with umap 240 30 0.5"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap240300_5.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap240300_5.pth'

echo "running with umap 240 30 1.5"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap240301_5.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap240301_5.pth'

echo "running with umap 300 30 0.5"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap300300_5.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap300300_5.pth'

echo "running with umap 300 30 1.5"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_umap300301_5.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_umap300301_5.pth'


echo "other"
echo "running with pca"
python3 ./main/train.py --inputs_path 'data/train_cite_inputs_PCA8061.csv' --targets_path 'data/train_cite_targets.csv' --model_path 'model/model_pca.pth'


