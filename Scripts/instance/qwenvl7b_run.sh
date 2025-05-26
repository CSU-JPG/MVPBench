#!/bin/bash
 python ../response.py \
 --data_dir '../Data/All_PhyTest/PhyTest' \
 --strategy 'CoT' \
 --config_path '../Configs/default.yaml' \
 --model_path '../../pretrained/MM-Eureka' \
 --output_dir '../Result/Dynamic_Prediction' \
 --max_tokens 1024 \
 --temperature 0.7 
