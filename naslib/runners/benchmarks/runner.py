import logging
import timeit
import os
import json

from naslib.predictors.zerocost import ZeroCost
from naslib.search_spaces import get_search_space
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils.get_dataset_api import get_dataset_api, load_sampled_architectures
from naslib.utils.logging import setup_logger
from naslib.utils import utils


def is_disconnected(net_str):
    ids = [int(i) for i in net_str.strip('()').split(',')]
    conv_ids = {2, 3}
    zero = 1

    edge_map = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    edge_map = {val: i for i, val in enumerate(edge_map)}

    def edge(i, j):
        return ids[edge_map[(i, j)]]

    # convolution is not connected to the output
    case_1 = (edge(3, 4) == zero) and (edge(2, 3) in conv_ids or edge(1, 3) in conv_ids)
    case_2 = (edge(1, 2) in conv_ids) and edge(2, 4) == zero and edge(3, 4) == zero
    case_3 = (edge(1, 2) in conv_ids) and edge(2, 4) == zero and edge(2, 3) == zero

    # convolution does not get any inputs
    case_4 = (edge(1, 2) == zero) and (edge(2, 3) in conv_ids or edge(2, 4) in conv_ids)
    case_5 = edge(1, 3) == zero and edge(2, 3) == zero and (edge(3, 4) in conv_ids)
    case_6 = edge(1, 2) == zero and edge(1, 3) == zero and (edge(3, 4) in conv_ids)
    return any([case_1, case_2, case_3, case_4, case_5, case_6])


def translate_str(s, replace_str='[]', with_str='()'):
    table = str.maketrans(replace_str, with_str)
    return str.translate(s, table)

config = utils.get_config_from_args()
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

search_space = get_search_space(config.search_space, config.dataset)
dataset_api = get_dataset_api(config.search_space, config.dataset)

if config.dataset in ['ninapro', 'svhn', 'scifar100']:
    postfix = '9x'
    with open(f'./naslib/data/9x/{config.search_space}/{config.dataset}/test.json') as f:
        api9x_data = json.load(f)
    api9x = {translate_str(str(record['arch'])): record['accuracy'] for record in api9x_data}
else:
    postfix = ''

archs = load_sampled_architectures(config.search_space, postfix)
end_index = (config.start_idx + config.n_models) if config.start_idx + config.n_models < len(archs) else len(archs)
archs_to_evaluate = {idx: eval(archs[str(idx)]) for idx in range(config.start_idx, end_index)}

utils.set_seed(config.seed)
train_loader, _, _, _, _ = utils.get_train_val_loaders(config)

config.normalize = config.normalize if hasattr(config, 'normalize') else 'no_norm'
config.relu = config.relu if hasattr(config, 'relu') else 'no_relu'
kwargs = {'normalize': config.normalize != 'no_norm', 'div_by_relu': config.relu != 'no_relu'}
predictor = ZeroCost(method_type=config.predictor, proxy_kwargs=kwargs)

zc_scores = []

for i, (idx, arch) in enumerate(archs_to_evaluate.items()):
    try:
        #if not is_disconnected(str(arch)):
        #    logger.info(f"{i} \tSkipping model id {idx} with encoding {arch}")
        #    continue

        logger.info(f'{i} \tComputing ZC score for model id {idx} with encoding {arch}')
        zc_score = {}
        graph = search_space.clone()
        graph.set_spec(arch)
        graph.parse()
        if config.dataset in ['ninapro', 'svhn', 'scifar100']:
            accuracy = api9x[str(arch)]
        else:
            accuracy = graph.query(Metric.VAL_ACCURACY, config.dataset, dataset_api=dataset_api)

        # Query predictor
        start_time = timeit.default_timer()
        score = predictor.query(graph, train_loader)
        end_time = timeit.default_timer()

        zc_score['idx'] = str(idx)
        zc_score['arch'] = str(arch)
        zc_score[predictor.method_type] = {
            'score': score,
            'time': end_time - start_time
        }
        zc_score['val_accuracy'] = accuracy
        zc_scores.append(zc_score)

        output_dir = os.path.join(config.data, 'zc_benchmarks', config.predictor)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f'benchmark--{config.normalize}--{config.relu}--{config.search_space}--{config.dataset}--{config.start_idx}.json')

        with open(output_file, 'w') as f:
            json.dump(zc_scores, f)
    except Exception as e:
        logger.info(f'Failed to compute score for model {idx} with arch {arch}!')

logger.info('Done.')
