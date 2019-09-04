# Author: bbrighttaer
# Project: ivpgan
# Date: 7/5/19
# Time: 8:31 AM
# File: bopt.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

from adgcca.hyper.base import ParamSearchAlg, ParamInstance
from adgcca.hyper.params import ConstantParam, LogRealParam, DiscreteParam, CategoricalParam, DictParam, \
    RealParam
from adgcca.utils.sim_data import DataNode
from adgcca.utils.train_helpers import save_model

size_suffix = "_size"


def _create_space(space, clazz, param_name, conf):
    def append(low, high, name):
        kwargs = {"low": low, "high": high, "name": name}
        if isinstance(conf, LogRealParam):
            kwargs["prior"] = "log-uniform"
        space.append(clazz(**kwargs))

    if clazz == Categorical:
        size = conf.size.max if isinstance(conf.size, DiscreteParam) else conf.size
        for i in range(size):
            space.append(Categorical(categories=conf.choices,
                                     name="{}___{}".format(param_name, i)))
    else:
        # if size is also a hyperparameter
        if isinstance(conf.size, DiscreteParam):
            # first add the size as a hyperparameter
            space.append(
                Integer(low=conf.size.min,
                        high=conf.size.max,
                        name="{}{}".format(param_name, size_suffix))
            )
            size = conf.size.max
        else:
            size = conf.size

        for i in range(size):
            low = pow(10, conf.min) if isinstance(conf, LogRealParam) else conf.min
            high = pow(10, conf.max) if isinstance(conf, LogRealParam) else conf.max
            append(low, high, "{}___{}".format(param_name, i))


def _transform_hparams_dict(params_config):
    return _to_skopt_space(params_config, [])


def _to_skopt_space(params_config, space, prefix=None):
    """
    converts the params config, in dict, to scikit-optimize space/dimension format

    Mapping:
    DiscreteParam --> skopt.space.Integer
    CategoricalParam --> skopt.space.Categorical
    LogRealParam --> skopt.space.Real
    RealParam --> skopt.space.Real

    Args:
    :param params_config: User-defined hyperparameter config used in creating the search algorithm.
    :param space: Used to aggregate all defined parameters transformed into skopt space objects.
    :param prefix: Used during recursion to identify sub-configurations and add transform them appropriately.
    :return: scikit-optimize compatible list of spaces to be used by `gp_minimize()`
    """
    for param, conf in zip(params_config.keys(), params_config.values()):
        clazz = None
        # Constant params are ignored in parameter search.
        if isinstance(conf, ConstantParam):
            continue

        if isinstance(conf, DiscreteParam):
            clazz = Integer
        elif isinstance(conf, CategoricalParam):
            clazz = Categorical
        elif isinstance(conf, LogRealParam) or isinstance(conf, RealParam):
            clazz = Real
        elif isinstance(conf, DictParam):
            _to_skopt_space(conf, space, prefix=param)

        if clazz:
            name = "{}__{}".format(prefix, param) if prefix else param
            _create_space(space, clazz, name, conf)
    return space


def _to_hparams_dict(bopt_params, params_config):
    return _convert_to_hparams(bopt_params, params_config, {})


def _convert_to_hparams(bopt_params, params_config, hparams, prefix=None):
    """
    Converts an skopt parameter set to a params dict usable in model training

    :param bopt_params: kwargs supplied by skopt
    :param params_config: The hyperparameter search config used in creating the search algorithm.
    :param hparams: used to collect the hyperparameters in recursion.
    :param prefix: used in recursion ops to identify sub-configs.
    :return: The hyperparameters in the same structure as `params_config` but with specific values for training.
    """
    for i, param in enumerate(params_config):
        conf = params_config[param]
        if isinstance(conf, ConstantParam):
            hparams[param] = conf.sample()
            continue
        elif isinstance(conf, DictParam):
            grp_dict = {}
            _convert_to_hparams(bopt_params, conf.p_dict, grp_dict, prefix="{}__".format(param))
            hparams[param] = grp_dict
            continue

        if isinstance(conf.size, DiscreteParam):
            size = bopt_params[param + size_suffix]
        else:
            size = conf.size

        val = []
        for j in range(size):
            prefix = prefix if prefix else ""
            val.append(bopt_params["{}{}___{}".format(prefix, param, j)])
        if not conf.is_list:
            val = val[0]
        hparams[param] = val

    return hparams


def _create_objective(alg, fold, train_data, val_data, model_dir, model_name, data_node, verbose=True):
    count = Count()
    iter_data_list = []
    data_node.data = iter_data_list

    @parse_config(params_config=alg.config)
    def objective(**bopt_params):
        count.inc()

        iter_data_node = DataNode(label="iteration-%d" % count.i)
        iter_data_list.append(iter_data_node)

        # Get hyperparameters.
        hparams = _to_hparams_dict(bopt_params=bopt_params, params_config=alg.config)

        if verbose:
            print("\nFold {}, param search iteration {}, hparams={}".format(fold, count.i, hparams))
        alg.stats.current_param = ParamInstance(hparams)

        # start of training with selected parameters
        best_model, score, epoch = train_model(hparams, iter_data_node)
        # end of training

        # get the score of this hyperparameter set
        score = alg.stats.current_param.score

        # save model
        if model_dir is not None and model_name is not None:
            save_model(best_model, model_dir,
                       "{}_{}-{}-fold{}-{}-{}-{}-{}-{:.4f}".format(alg.dataset_label, alg.view,
                                                                   alg.stats.current_param.id,
                                                                   fold, count.i, model_name, alg.split_label, epoch,
                                                                   score))

        if verbose:
            print("BayOpt hparams search iter = {}: params = {}".format(count.i, alg.stats.current_param))

        # move current hparams to records
        alg.stats.update_records()

        # avoid nan scores in search. TODO: replace this hack with an organic approach.
        if str(score) == "nan":
            score = -1e5

        # we want to maximize the score so negate it to invert minimization by skopt
        return -score

    def train_model(hparams, sim_data_node):
        # initialize model, dataloaders, and other elements.
        init_objs = alg.initializer_fn(hparams, train_data, val_data, **alg.init_args)

        # model training
        alg.train_args["sim_data_node"] = sim_data_node
        best_model, score, epoch = alg.train_fn(alg._score_fn, *init_objs, **alg.train_args)
        return best_model, score, epoch

    return objective


def parse_config(params_config):
    bopt_space = _transform_hparams_dict(params_config)
    decorator = use_named_args(bopt_space)
    return decorator


class Count(object):
    def __init__(self):
        self.i = -1

    def inc(self):
        self.i += 1


class BayesianOptSearchCV(ParamSearchAlg):

    def fit(self, model_dir, model_name, max_iter=100, verbose=True, seed=None):
        folds_data = []
        self.data_node.data = folds_data
        for fold in range(self.num_folds):
            k_node = DataNode(label="BayOpt_search_fold-%d" % fold)
            folds_data.append(k_node)

            # Get data
            data = self.data_provider_fn(fold, **self.data_args)
            train_data = data["train"]
            val_data = data["val"]
            if "test" in data:
                test_data = data["test"]
                self.init_args["test_dataset"] = test_data

            # BayOpt hyperparameter search.
            space = _transform_hparams_dict(self.config)
            print("BayOpt space dimension=%d" % len(space))
            res_gp = gp_minimize(
                func=_create_objective(self, fold, train_data, val_data, model_dir, model_name, k_node, verbose),
                dimensions=space,
                n_calls=max_iter,
                random_state=seed,
                acq_func="EI")

            print("Fold {}, best score={:.4f}".format(fold, res_gp.fun))

        return self.stats