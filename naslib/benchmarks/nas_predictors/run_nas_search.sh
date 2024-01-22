optimizer=$5

predictors=(bananas mlp lgb gcn xgb ngb rf dngo \
bohamiann bayes_lin_reg seminas nao gp sparse_gp var_sparse_gp)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
outdir=search_nb101_$optimizer

# search space / data:
search_space=$6
dataset=$7
search_epochs=500

# trials / seeds:
trials=$4
end_seed=$(($start_seed + $trials - 1))

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    if [[ $8 ]]; then
         if [ $predictor != $8 ]; then
             continue;
         fi
    fi

    graph_pred_args="--graph_features_pickle_path $2 --valid_networks $3"

    python create_configs.py --predictor $predictor \
    --epochs $search_epochs --start_seed $start_seed --trials $trials \
    --out_dir $out_dir --dataset=$dataset --config_type nas_predictor \
    --search_space $search_space --optimizer $optimizer \
    $graph_pred_args
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
    for predictor in ${predictors[@]}
    do
        config_file=$out_dir/$dataset/configs/nas_predictors/config\_$optimizer\_$predictor\_$t.yaml
        echo ================running $predictor trial: $t =====================
        python nas_predictors/runner.py --config-file $config_file
    done
done
