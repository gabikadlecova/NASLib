import json
import pickle

import numpy as np
from naslib.predictors.trees.ngb import loguniform
from zc_combine.predictors import predictor_cls

from naslib.predictors import Predictor
from zc_combine.utils.script_utils import load_feature_proxy_dataset, create_cache_filename


def replace_path_root(cfg_dict, root_path, root_str="<ROOT>"):
    for k in cfg_dict.keys():
        if isinstance(cfg_dict[k], str):
            cfg_dict[k] = cfg_dict[k].replace(root_str, root_path)


class GraphFeaturesPredictor(Predictor):
    def __init__(self, config, hpo_wrapper=False):
        super().__init__()

        cgf_path = config.graph_features_config_path
        with open(cgf_path, 'r') as f:
            self.cfg = json.load(f)

        replace_path_root(self.cfg, config.zc_root_path)
        replace_path_root(self.cfg['kwargs'], config.zc_root_path)

        self.model = predictor_cls[self.cfg['model']](config.seed)
        self.bench_name = f"zc_{config.search_space}"
        self.dataset = config.dataset
        self.init_dataset()

        self.hpo_wrapper = hpo_wrapper

    def init_dataset(self):
        if 'cache_dir' in self.cfg:
            if 'use_features' not in self.cfg['kwargs'] or self.cfg['kwargs']['use_features']:
                kwargs = self.cfg['kwargs']
                self.cfg['kwargs']['cache_path'] = create_cache_filename(self.cfg['cache_dir'],
                                                                         kwargs['cfg'],
                                                                         kwargs.get('features', None),
                                                                         kwargs.get('version_key', None),
                                                                         True)

        nets, dataset, y = load_feature_proxy_dataset(self.cfg['searchspace_path'], self.bench_name, self.dataset,
                                                      **self.cfg['kwargs'])

        self.net_map = {nets.loc[i]: i for i in nets.index}
        self.dataset = dataset
        self.y = y

    def save_dataset(self, out_path):
        with open(out_path, 'wb') as f:
            pickle.dump((self.net_map, self.dataset, self.y), f)

    def get_features_for_net(self, x_arch):
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
        # NOTE: Copied from NB301
        params = {
            "n_estimators": 116,
            "max_features": 0.17055852159745608,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "bootstrap": False,
            #'verbose': -1
        }
        return params

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            # evaluate the default config first during HPO
            params = self.inital_hyperparams.copy()
        else:
            params = {
                "n_estimators": int(loguniform(16, 128)),
                "max_features": loguniform(0.1, 0.9),
                "min_samples_leaf": int(np.random.choice(19) + 1),
                "min_samples_split": int(np.random.choice(18) + 2),
                "bootstrap": False,
                #'verbose': -1
            }
        self.hyperparams = params
        return params
