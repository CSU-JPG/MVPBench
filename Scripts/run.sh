#!/bin/bash
 python ../response.py \
 --data_dir '../Data/PhyTest' \
 --strategy 'CoT' \
 --config_path '../Configs/default.yaml' \
 --model_path '../pretrained/InternVL2_5-78B' \
 --output_dir 'Result/PhyTest' \
 --max_tokens 1024 \
 --temperature 0.7 
 # --subject 'PhyTest' 'FhysExperiment' 'PhySpatial' 'DynamicPrediction' \

