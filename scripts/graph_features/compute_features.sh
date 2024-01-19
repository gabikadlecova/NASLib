#!/bin/bash


out_dir=$1
dataset=$2
search_space=$3

graph_pred_args="--graph_features_config_path $4 --graph_features_path $5"

if [[ $6 ]]; then
  graph_pred_args="$graph_pred_args --valid_networks $6"
fi

echo $graph_pred_args

python3 ../create_configs.py --predictor graph_features --experiment_type vary_train_size \
  --test_size 200 --start_seed 0 --trials 1 --out_dir $out_dir \
  --dataset=$dataset --config_type predictor --search_space $search_space \
  $graph_pred_args

python3 save_graph_features.py --config-file $out_dir/$dataset/configs/predictors/config_graph_features_0.yaml \
  --out_path "$out_dir"$search_space-$dataset.pickle

rm $out_dir/$dataset/configs/predictors/config_graph_features_0.yaml

