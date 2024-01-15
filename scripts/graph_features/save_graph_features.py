from naslib.predictors.graph_features import GraphFeaturesPredictor
from naslib import utils
from naslib.utils import default_argument_parser


def save_predictor(args):
    config = utils.get_config_from_args(args, config_type="predictor")
    pred = GraphFeaturesPredictor(config)
    print(args.out_path)
    pred.save_dataset(args.out_path)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--out_path', type=str, help='Pickle path to save graph features')
    args = parser.parse_args()
    save_predictor(args)
