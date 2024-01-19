import json
import pickle

import numpy as np
from naslib.predictors.trees.ngb import loguniform
from zc_combine.predictors import predictor_cls

from naslib.predictors import Predictor


def replace_path_root(cfg_dict, root_path, root_str="<ROOT>"):
    for k in cfg_dict.keys():
        if isinstance(cfg_dict[k], str):
            cfg_dict[k] = cfg_dict[k].replace(root_str, root_path)


class GraphFeaturesPredictor(Predictor):
    def __init__(self, config, model='rf', hpo_wrapper=False):
        super().__init__()

        feature_path = config.graph_features_pickle_path
        with open(feature_path, 'rb') as f:
            data = pickle.load(f)
            self.net_map, self.dataset, self.y = data

        self.dataset.drop(columns='net', inplace=True)

        self.model_name = model
        self.model = predictor_cls[model](config.seed)
        self.bench_name = f"zc_{config.search_space}"

        self.hpo_wrapper = hpo_wrapper

    def get_features_for_net(self, x_arch):
        # ugly hack for padded vs not padded issues (nb101) in old NASLib version
        if hasattr(x_arch[0], 'hash_not_padded'):
            hashes = [str(arch.hash_not_padded) for arch in x_arch]
        else:
            hashes = [str(arch.get_hash()) for arch in x_arch]

        indices = [self.net_map[h] for h in hashes]
        return self.dataset.loc[indices], self.y.loc[indices]

    def fit(self, xtrain, ytrain, train_info=None):
        x, y_dataset = self.get_features_for_net(xtrain)
        self.model.fit(x, ytrain)

        train_pred = self.query(xtrain)
        train_error = np.mean(abs(train_pred - ytrain))
        return train_error

    def query(self, xtest, info=None):
        x, y_dataset = self.get_features_for_net(xtest)
        return self.model.predict(x)

    @property
    def inital_hyperparams(self):
        if self.model == 'rf':
            # NOTE: Copied from NB301
            params = {
                "n_estimators": 116,
                "max_features": 0.17055852159745608,
                "min_samples_leaf": 2,
                "min_samples_split": 2,
                "bootstrap": False,
                #'verbose': -1
            }
        elif self.model == 'xgb':
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': "rmse",
                'booster': 'gbtree',
                'max_depth': 6,
                'min_child_weight': 1,
                'colsample_bytree': 1,
                'learning_rate': .3,
                'colsample_bylevel': 1
            }
        else:
            raise ValueError("Invalid model for hyperparams")

        return params

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            # evaluate the default config first during HPO
            params = self.inital_hyperparams.copy()
        else:
            if self.model == 'rf':
                params = {
                    "n_estimators": int(loguniform(16, 128)),
                    "max_features": loguniform(0.1, 0.9),
                    "min_samples_leaf": int(np.random.choice(19) + 1),
                    "min_samples_split": int(np.random.choice(18) + 2),
                    "bootstrap": False,
                    #'verbose': -1
                }
            elif self.model == 'xgb':
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': "rmse",
                    # 'early_stopping_rounds': 100,
                    'booster': 'gbtree',
                    'max_depth': int(np.random.choice(range(1, 15))),
                    'min_child_weight': int(np.random.choice(range(1, 10))),
                    'colsample_bytree': np.random.uniform(.0, 1.0),
                    'learning_rate': loguniform(.001, .5),
                    # 'alpha': 0.24167936088332426,
                    # 'lambda': 31.393252465064943,
                    'colsample_bylevel': np.random.uniform(.0, 1.0),
                }
            else:
                raise ValueError("Invalid model for hyperparams")

        self.hyperparams = params
        return params
