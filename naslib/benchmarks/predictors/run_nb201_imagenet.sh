predictors=(fisher grad_norm grasp jacov snip synflow \
lce lce_m sotl sotle valacc valloss \
lcsvr omni_ngb omni_seminas \
bananas bonas gcn mlp nao seminas \
lgb ngb rf xgb \
bayes_lin_reg bohamiann dngo \
gp sparse_gp var_sparse_gp \
graph_features graph_features_xgb graph_features_tune graph_features_xgb_tune)

experiment_types=(single single single single single single \
vary_fidelity vary_fidelity vary_fidelity vary_fidelity vary_fidelity vary_fidelity \
vary_both vary_both vary_both \
vary_train_size vary_train_size vary_train_size vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size vary_train_size)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
out_dir=runs

# search space / data:
search_space=nasbench201
dataset=ImageNet16-120

# other variables:
trials=$4
end_seed=$(($start_seed + $trials - 1))
test_size=200

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    experiment_type=${experiment_types[$i]}

    if [[ $5 ]]; then
         if [ $predictor != $5 ]; then
             continue;
         fi
    fi

    graph_pred_args="--graph_features_pickle_path $2 --valid_networks $3"

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
        if [[ $5 ]]; then
             if [ $predictor != $5 ]; then
                 continue;
             fi
        fi

        config_file=$out_dir/$dataset/configs/predictors/config\_$predictor\_$t.yaml
        echo ================running $predictor trial: $t =====================
        python predictors/runner.py --config-file $config_file
    done
done
