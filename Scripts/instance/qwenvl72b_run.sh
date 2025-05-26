#!/bin/bash
python ../response.py \
 --data_dir '../Data/DynamicPrediction' \
 --strategy 'CoT' \
 --config_path '../Configs/default.yaml' \
 --model_path '../../../zhuobai/qwen_2.5_vl_72B/qwen2_5vl-72B' \
 --output_dir '../Result/Dynamic_Prediction' \
 --max_tokens 1024 \
 --temperature 0.7 
