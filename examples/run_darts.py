import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch
from naslib.search_spaces import DartsSearchSpace, SimpleCellSearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args

config = get_config_from_args()  # use --help so see the options
set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is very verbose

search_space = DartsSearchSpace()  # use SimpleCellSearchSpace() for less heavy search

optimizer = DARTSOptimizer(**config.search)
optimizer.adapt_search_space(search_space, config.dataset) # Should this call be optim.adapt_s_s(s_s, config.dataset)?

trainer = Trainer(optimizer, config)
trainer.search()  # Search for an architecture
trainer.evaluate()  # Evaluate the best architecture
