# Author: bbrighttaer
# Project: ivpgan
# Date: 7/5/19
# Time: 11:58 AM
# File: base.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
from datetime import datetime as dt

import numpy as np
import pandas as pd

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")


class ParamSearchAlg(abc.ABC):
    """Performs a random search for hyperparameters.

    Args:
    ---------
        hparam_config: dict
            A dictionary of hyperparameter configurations for the search.
        num_folds: int
            The number of data folds to be used for the search.
        initializer: callable
            The function to be called for initializing the model and all elements for training. The hyperparameters
            generated using the passed configuration shall be passed first to this function and the next two parameters
            shall be the train and validation data, respectively, as provided by the `data_provider` function.
        data_provider: callable
            This function shall be called to provide the train and validation data for each fold of the search. Hence,
            the first parameter of this function accepts the current fold number. It shall return a dict with keys:
            {'train':data, 'val':data}. The returned datasets are passed to the initializer to create dataloaders.
        train_fn: callable
            The function that implements model training. All return values of the initializer function are passed (in
            the order returned) to this function after `eval_fn` and then followed by any other extra arguments in
            `train_args`.
            This function shall return the best model after the training.
        eval_fn: callable
            A function that is passed to `train_fn` to evaluate the performance of a trained model and return the
            score for the hyperparameters used for the training. It must returned a single real-value.
            It's assumed this evaluation function is called after every validation batch.
        init_args: dict
            A dictionary of keyword-args (kwargs) that are extra arguments to be passed to the initialization function
            if any.
        data_args: dict
            A dictionary of `kwargs` that are passed as extra arguments to the `data_provider` function after the
            current fold number (fold number counting starts at 0).
        train_args: dict
            A dictionary of `kwargs` that are passed to `train_fn` after all values of the `initializer` function have
            been passed.
        data_node: ``DataNode`` for collecting a simulation instance's data
    """

    def __init__(self, hparam_config, num_folds, initializer, data_provider, train_fn, eval_fn,
                 init_args=dict(), data_args=dict(), train_args=dict(), data_node=None, split_label=None,
                 view_label=None, dataset_label=None):
        self.config = hparam_config
        self.num_folds = num_folds
        self.initializer_fn = initializer
        self.data_provider_fn = data_provider
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.init_args = init_args
        self.data_args = data_args
        self.train_args = train_args

        self.data_node = data_node
        self.split_label = split_label
        self.view = view_label
        self.dataset_label = dataset_label

        self.stats = HyperParamStats()

    def _score_fn(self, *args, **kwargs):
        score = self.eval_fn(*args, **kwargs)
        self.stats.current_param.add_score(score)
        return score

    @abc.abstractmethod
    def fit(self, model_dir, model_name, max_iter=5, verbose=True):
        pass


class ParamInstance(object):
    """Keeps information of a set of parameters used for the training algorithm at a point in time."""

    def __init__(self, hparams, id=None):
        self._id = date_label if id is None else id
        self._hparams = hparams
        self._score = []

    @property
    def id(self):
        return self._id

    @property
    def params(self):
        return self._hparams

    @property
    def score(self):
        return np.mean(self._score)

    def add_score(self, s):
        self._score.append(s)

    def as_dict(self):
        return {**self.params, "score": self.score}

    def __repr__(self):
        return str(self.as_dict())  # "[id={} : hparams={} : score={}]".format(self._id, self._hparams, self.score)


class HyperParamStats(object):
    """
    Keeps records of hyperparameter instances: :class:`ParamInstance`
    """

    def __init__(self):
        self.records = []
        self.current_param = None

    def update_records(self):
        if self.current_param is not None:
            # Ignore hparam sets without a score (they led to nans so no score)ss
            scores = self.current_param._score
            if len(scores) > 0 and str(scores[0]) != "nan":
                self.records.append(self.current_param.as_dict())
            self.current_param = None

    def __repr__(self):
        s = ""
        for rec in self.records:
            s += str(rec) + "\n"
        return s

    def best(self, m="min"):
        """
        Find the hyperparameter set in the records that has the best performance. The default approach is that, low
        values are better.

        :param m: Indicates whether low or high values are better. Valid values are 'min' and 'max'
        :return: The best hyperparameter set.
        """
        params = None
        best = None
        for rec in self.records:
            if not best:
                best = rec["score"]
                params = rec
                continue

            if m == "max":
                if rec["score"] > best:
                    best = rec["score"]
                    params = rec
            else:
                if rec["score"] < best:
                    best = rec["score"]
                    params = rec
        return params

    def to_csv(self, file):
        """
        Creates a :class:`pandas.DataFrame` out of the recorded hyperparameters and save to a file as csv.

        :param file:
        :return:
        """
        df_dict = dict()
        for rec in self.records:
            for k in rec:
                if k not in df_dict:
                    df_dict[k] = [rec[k]]
                else:
                    df_dict[k].append(rec[k])
        df = pd.DataFrame(data=df_dict)
        df = df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
        df.to_csv(file, index=False)
