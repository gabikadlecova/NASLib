import glob
import json
import os

import click
import pandas as pd


@click.command()
@click.argument('out_csv')
@click.argument('runs_dir')
@click.option('--dataset_values', default='cifar10,ImageNet16-120', help="Possible dataset subdirs.")
def main(out_csv, runs_dir, dataset_values):
    if os.path.exists(out_csv):
        print(f"File {out_csv} already exists. Do you want to overwrite it? (y/n)")
        ans = input()
        if ans != 'y':
            print("Exiting.")
            return

    dataset_values = dataset_values.split(',')
    dataset = runs_dir.split('/')[1]

    assert dataset in dataset_values, f"Dataset {dataset} not in {dataset_values}."

    results = []

    for predictor_dir in glob.glob(f"{runs_dir}/*"):
        predictor = predictor_dir.split('/')[-1]

        for seed in os.listdir(predictor_dir):
            errors_path = f"{predictor_dir}/{seed}/errors.json"
            if not os.path.exists(errors_path):
                print(f"File {errors_path} does not exist, skipping.")
                continue

            with open(errors_path, 'r') as f:
                 data = json.load(f)

            for entry in data[1:]:
                res = {'predictor': predictor, 'seed': seed, 'dataset': dataset, 'train_size': entry['train_size']}
                metrics = {k: entry[k] for k in ['kendalltau', 'spearman', 'train_time', 'fit_time']}

                results.append({**res, **metrics})

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)


if __name__ == '__main__':
    main()
