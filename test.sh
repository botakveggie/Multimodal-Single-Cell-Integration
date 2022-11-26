#echo "using 200 30 1 model"

#echo "using full data"
#python3 main/get_predictions.py --inputs_path "gse/test_full_inputs_200301.csv" --model_path "model/model_200301_final.pth" --output_path "output/output_200301.csv"

#echo "using red data"
#python3 main/get_predictions.py --inputs_path "gse/test_red_inputs_200301.csv" --model_path "model/model_200301_final.pth" --output_path "output/output_200301_red.csv"



#echo "using 240 50 1 33  model"

#echo "using full data"
#python3 main/get_predictions.py --inputs_path "gse/test_full_inputs_240501.csv" --model_path "model/model_240501_final_33.pth" --output_path "output/output_240501_33.csv"

#echo "using red data"
#python3 main/get_predictions.py --inputs_path "gse/test_red_inputs_240501.csv" --model_path "model/model_240501_final_33.pth" --output_path "output/output_240501_33r.csv"



#echo "using 240 50 1 30  model"

#echo "using full data"
#python3 main/get_predictions.py --inputs_path "gse/test_full_inputs_240501.csv" --model_path "model/model_240501_final_30.pth" --output_path "output/output_240501_30.csv"

#echo "using red data"
#python3 main/get_predictions.py --inputs_path "gse/test_red_inputs_240501.csv" --model_path "model/model_240501_final_30.pth" --output_path "output/output_240501_30r.csv"


echo "getting correlation score"

echo "using 200 30 1 model"
echo "full"
python3 gse/get_score.py --preds_path "output/output_200301.csv" --outputs_path "gse/test_targets.csv"
echo "red"
python3 gse/get_score.py --preds_path "output/output_200301_red.csv" --outputs_path "gse/test_targets.csv"

echo ""
echo "using 240 50 1 model 30"
echo "full"
python3 gse/get_score.py --preds_path "output/output_240501_30.csv" --outputs_path "gse/test_targets.csv"
echo "red"
python3 gse/get_score.py --preds_path "output/output_240501_30r.csv" --outputs_path "gse/test_targets.csv"

echo ""
echo "using 240 50 1 model 33"
echo "full"
python3 gse/get_score.py --preds_path "output/output_240501_33.csv" --outputs_path "gse/test_targets.csv"
echo "red"
python3 gse/get_score.py --preds_path "output/output_240501_33r.csv" --outputs_path "gse/test_targets.csv"
