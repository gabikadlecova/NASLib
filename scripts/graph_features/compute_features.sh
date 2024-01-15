#!/bin/bash


out_dir=`basename $1`
dataset=$2
search_space=$3

graph_pred_args="--graph_features_config_path $2 --graph_features_path $3 --valid_networks $4"

python create_configs.py --predictor graph_features --experiment_type vary_train_size \
  --test_size 200 --start_seed 0 --trials 1 --out_dir $out_dir \
  --dataset=$dataset --config_type predictor --search_space $search_space \
  $graph_pred_args

python save_graph_features.py --config_path $out_dir/$dataset/configs/predictors/config_graph_features_0.yaml \
  --out_path $1

rm $out_dir/$dataset/configs/predictors/config_graph_features_0.yaml
