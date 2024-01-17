predictors=(fisher grad_norm grasp jacov snip synflow \
bananas bonas gcn mlp nao seminas \
lgb ngb rf xgb \
bayes_lin_reg bohamiann dngo \
gp sparse_gp var_sparse_gp \
graph_features)

experiment_types=(single single single single single single \
vary_train_size vary_train_size vary_train_size vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size \
vary_train_size)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
out_dir=runs

# search space / data:
search_space=nasbench101
dataset=cifar10

# other variables:
trials=1
end_seed=$(($start_seed + $trials - 1))
test_size=200

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    experiment_type=${experiment_types[$i]}

    graph_pred_args="--graph_features_pickle_path $2 --graph_features_model $3"

    python create_configs.py --predictor $predictor --experiment_type $experiment_type \
    --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --config_type predictor --search_space $search_space \
    $graph_pred_args
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
    for predictor in ${predictors[@]}
    do
        config_file=$out_dir/$dataset/configs/predictors/config\_$predictor\_$t.yaml
        echo ================running $predictor trial: $t =====================
        python $base_file/benchmarks/predictors/runner.py --config-file $config_file
    done
done
