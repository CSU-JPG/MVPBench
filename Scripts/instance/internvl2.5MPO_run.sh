#!/bin/bash
python ../response.py \
 --data_dir '../Data/DynamicPrediction' \
 --strategy 'CoT' \
 --config_path '../Configs/default.yaml' \
 --model_path '../../pretrained/InternVL2_5-78B-MPO' \
 --output_dir '../Result/Dynamic_Prediction' \
 --max_tokens 1024 \
 --temperature 0.7 
 
