# Author: bbrighttaer
# Project: ivpgan
# Date: 7/5/19
# Time: 1:49 PM
# File: test.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from skopt.space import Integer, Real, Categorical

from ivpgan.hyper.params import ConstantParam, LogRealParam, DiscreteParam, CategoricalParam, DictParam, \
    RealParam

size_suffix = "_size"
# grp_prefix = "group_"


def get_hparam_config():
    num_active_views = 3
    return {
        # global params
        "latent_dim": ConstantParam(c=49),
        "detach": ConstantParam("detach"),
        "precision": ConstantParam(10000),
        "use_task_grad": ConstantParam(True),
        "view_weights": LogRealParam(min=-1, size=num_active_views),
        "hdims": DiscreteParam(min=256, max=5000, size=DiscreteParam(min=1, max=4)),

        # weight initialization
        "kaiming_constant": DiscreteParam(min=2, max=10),


        # parameter groups learning rates
        "pre_attn_lr": LogRealParam(),
        "attn_lr": LogRealParam(),
        "rep_lr": LogRealParam(),
        "task_lr": LogRealParam(),

        # dropout regs
        "rep_dropout": LogRealParam(min=-2),
        "attn_dropout": LogRealParam(min=-2),
        "task_dropout": ConstantParam(0),
        "pre_attn_dropout": LogRealParam(min=-2),

        "tr_batch_size": DiscreteParam(min=500, max=1500),
        "val_batch_size": ConstantParam(512),
        "test_batch_size": ConstantParam(512),

        # optimizer params
        "optimizer": CategoricalParam(choices=["sgd", "adam"]),
        "optimizer__sgd__weight_decay": LogRealParam(),
        "optimizer__sgd__nesterov": CategoricalParam(choices=[True, False]),
        "optimizer__sgd__momentum": LogRealParam(min=-1),
        "optimizer__adam__weight_decay": LogRealParam(),

        # ECFP params
        "ecfp_params": DictParam({
            "param1": DiscreteParam(min=10, max=50),
            "radius": ConstantParam(c=1024),
            "param2": RealParam(min=1, max=2)
        }),

        # Weave params
        "weave_params": DictParam({
            "param1": DiscreteParam(min=10, max=50),
            "output_dim": ConstantParam(128),
            "param2": RealParam(min=1, max=2)
        }),

        # Graph conv params
        "gconv_params": DictParam({
            "param1": DiscreteParam(min=10, max=50),
            "output_dim": ConstantParam(128),
            "param2": RealParam(min=1, max=2)
        }),
    }


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


def _transform_hparams_dict(params_config, space=[], prefix=None):
    """converts the params config, in dict, to scikit-optimize space/dimension format

    Mapping:
    DiscreteParam --> skopt.space.Integer
    CategoricalParam --> skopt.space.Categorical
    LogRealParam --> skopt.space.Real
    RealParam --> skopt.space.Real
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
            _transform_hparams_dict(conf, prefix=param)

        if clazz:
            name = "{}__{}".format(prefix, param) if prefix else param
            _create_space(space, clazz, name, conf)
    return space


def _to_hparams_dict(bopt_params, params_config, hparams={}, prefix=None):
    """converts an skopt parameter set to a params dict usable in model training"""
    for i, param in enumerate(params_config):
        conf = params_config[param]
        if isinstance(conf, ConstantParam):
            hparams[param] = conf.sample()
            continue
        elif isinstance(conf, DictParam):
            grp_dict = {}
            _to_hparams_dict(bopt_params, conf.p_dict, grp_dict, prefix="{}__".format(param))
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
        val = val[0] if len(val) == 1 else val
        hparams[param] = val

    return hparams


if __name__ == '__main__':
    config = get_hparam_config()
    space = _transform_hparams_dict(config)
    for s in space:
        print(s.name, s)

    # fake sample
    params = {}
    for s in space:
        params[s.name] = np.random.randint(0, 2)

    print(params)

    params = _to_hparams_dict(params, config)
    print(params)
