import argparse

from naslib.predictors.graph_features import GraphFeaturesPredictor
from naslib import utils


def save_predictor(args):
    config = utils.get_config_from_args(args, config_type="predictor")
    pred = GraphFeaturesPredictor(config)
    pred.save_dataset(args['out_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save graph features')
    parser.add_argument('--out_path', type=str, help='Pickle path to save graph features')
    parser.add_argument('--config_path', type=str, help='Pickle path to save graph features')
    args = parser.parse_args()
    save_predictor(args)
