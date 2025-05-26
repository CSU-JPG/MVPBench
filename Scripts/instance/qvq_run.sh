#!/bin/bash
 python ../response.py \
 --data_dir '../Data/DynamicPrediction' \
 --strategy 'CoT' \
 --config_path '../Configs/default.yaml' \
 --model_path '../../pretrained/QVQ' \
 --output_dir '../Result/Dynamic_Prediction' \
 --max_tokens 512 \
 --temperature 0.7 


