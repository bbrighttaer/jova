# Author: bbrighttaer
# Project: jova
# Date: 7/2/19
# Time: 1:24 PM
# File: kronrls.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
from datetime import datetime as dt

import numpy as np
from soek import RealParam
from soek.bopt import BayesianOptSearchCV
from soek.params import ConstantParam
from soek.rand import RandomSearchCV

import jova.metrics as mt
from jova.data import get_data, load_proteins, DtiDataset
from jova.metrics import compute_model_performance
from jova.utils import Trainer, io
from jova.utils.io import save_model
from jova.utils.sim_data import DataNode

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

# seeds = [123, 124, 125]
seeds = [1, 8, 64]


class KronRLS(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset, drug_kernel_dict, prot_kernel_dict):
        data = None
        # metrics
        metrics = [mt.Metric(mt.rms_score, np.nanmean),
                   mt.Metric(mt.concordance_index, np.nanmean),
                   mt.Metric(mt.pearson_r2_score, np.nanmean)]
        return data, hparams['reg_lambda'], metrics

    @staticmethod
    def data_provider(fold, flags, data_dict):
        if not flags['cv']:
            print("Training scheme: train, validation" + (", test split" if flags['test'] else " split"))
            train_dataset = DtiDataset(x_s=[data[1][0].X for data in data_dict.values()],
                                       y_s=[data[1][0].y for data in data_dict.values()],
                                       w_s=[data[1][0].w for data in data_dict.values()])
            valid_dataset = DtiDataset(x_s=[data[1][1].X for data in data_dict.values()],
                                       y_s=[data[1][1].y for data in data_dict.values()],
                                       w_s=[data[1][1].w for data in data_dict.values()])
            test_dataset = None
            if flags['test']:
                test_dataset = DtiDataset(x_s=[data[1][2].X for data in data_dict.values()],
                                          y_s=[data[1][2].y for data in data_dict.values()],
                                          w_s=[data[1][2].w for data in data_dict.values()])
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset}
        else:
            train_dataset = DtiDataset(x_s=[data[1][fold][0].X for data in data_dict.values()],
                                       y_s=[data[1][fold][0].y for data in data_dict.values()],
                                       w_s=[data[1][fold][0].w for data in data_dict.values()])
            valid_dataset = DtiDataset(x_s=[data[1][fold][1].X for data in data_dict.values()],
                                       y_s=[data[1][fold][1].y for data in data_dict.values()],
                                       w_s=[data[1][fold][1].w for data in data_dict.values()])
            test_dataset = DtiDataset(x_s=[data[1][fold][2].X for data in data_dict.values()],
                                      y_s=[data[1][fold][2].y for data in data_dict.values()],
                                      w_s=[data[1][fold][2].w for data in data_dict.values()])
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset}
        return data

    @staticmethod
    def evaluate(eval_dict, y, y_pred, w, metrics, tasks, transformers):
        eval_dict.update(compute_model_performance(metrics, y_pred.cpu().detach().numpy(), y, w, transformers,
                                                   tasks=tasks))
        # scoring
        rms = np.nanmean(eval_dict["nanmean-rms_score"])
        ci = np.nanmean(eval_dict["nanmean-concordance_index"])
        r2 = np.nanmean(eval_dict["nanmean-pearson_r2_score"])
        score = np.nanmean([ci, r2]) - rms
        return score

    @staticmethod
    def train(data, reg_lambda, metrics, transformer, drug_kernel_dict, prot_kernel_dict, sim_data_node,
              is_hsearch=False):
        print('Inside train func')

    @staticmethod
    def evaluate_model():
        pass


def main(flags):
    if len(flags["views"]) > 0:
        print("Single views for training: {}, num={}".format(flags["views"], len(flags["views"])))
    else:
        print("No views selected for training")

    for view in flags["views"]:
        cview, pview = view
        sim_label = "single_view_{}_{}".format(cview, pview)
        print(sim_label)

        # Simulation data resource tree
        split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
            flags["cold_drug"] else "None"
        dataset_lbl = flags["dataset"]
        node_label = "{}_{}_{}_{}_{}_{}".format(dataset_lbl, cview, pview, split_label,
                                                "eval" if flags["eval"] else "train", date_label)
        sim_data = DataNode(label=node_label)
        nodes_list = []
        sim_data.data = nodes_list

        prot_desc_dict, prot_seq_dict = load_proteins(flags['prot_desc_path'])

        # For searching over multiple seeds
        hparam_search = None

        for seed in seeds:
            # for data collection of this round of simulation.
            data_node = DataNode(label="seed_%d" % seed)
            nodes_list.append(data_node)

            random.seed(seed)
            np.random.seed(seed)

            # load data
            print('-------------------------------------')
            print('Running on dataset: %s' % dataset_lbl)
            print('-------------------------------------')

            data_dict = dict()
            data_key = {"ecfp4": "KRLS_ECFP4",
                        "ecfp8": "KRLS_ECFP8",
                        "weave": "KRLS_Weave",
                        "gconv": "KRLS_GraphConv",
                        "gnn": "KRLS_GNN"}.get(cview)
            data_dict[cview] = get_data(data_key, flags, prot_sequences=prot_seq_dict, seed=seed)
            transformer = data_dict[cview][2]
            drug_kernel_dict, prot_kernel_dict = data_dict[cview][4]
            tasks = data_dict[cview][0]
            flags["tasks"] = tasks

            trainer = KronRLS()

            if flags["cv"]:
                k = flags["fold_num"]
                print("{}, {}-{}: Training scheme: {}-fold cross-validation".format(tasks, cview, pview, k))
            else:
                k = 1
                print("{}, {}-{}: Training scheme: train, validation".format(tasks, cview, pview)
                      + (", test split" if flags['test'] else " split"))

            if flags["hparam_search"]:
                print("Hyperparameter search enabled: {}".format(flags["hparam_search_alg"]))

                # arguments to callables
                extra_init_args = {}
                extra_data_args = {"flags": flags,
                                   "data_dict": data_dict}
                n_iters = 3000
                extra_train_args = {"transformer": transformer,
                                    "drug_kernel_dict": drug_kernel_dict,
                                    "prot_kernel_dict": prot_kernel_dict,
                                    "is_hsearch": True}

                hparams_conf = get_hparam_config(flags, cview, pview)

                if hparam_search is None:
                    search_alg = {"random_search": RandomSearchCV,
                                  "bayopt_search": BayesianOptSearchCV}.get(flags["hparam_search_alg"],
                                                                            BayesianOptSearchCV)
                    min_opt = "gp"
                    hparam_search = search_alg(hparam_config=hparams_conf,
                                               num_folds=k,
                                               initializer=trainer.initialize,
                                               data_provider=trainer.data_provider,
                                               train_fn=trainer.train,
                                               save_model_fn=io.save_model,
                                               init_args=extra_init_args,
                                               data_args=extra_data_args,
                                               train_args=extra_train_args,
                                               data_node=data_node,
                                               split_label=split_label,
                                               sim_label=sim_label,
                                               minimizer=min_opt,
                                               dataset_label=dataset_lbl,
                                               results_file="{}_{}_dti_{}_{}_{}.csv".format(
                                                   flags["hparam_search_alg"], sim_label, date_label, min_opt, n_iters))

                stats = hparam_search.fit(model_dir="models", model_name="".join(tasks), max_iter=20, seed=seed)
                print(stats)
                print("Best params = {}".format(stats.best(m="max")))
            else:
                invoke_train(trainer, tasks, data_dict, transformer, flags, data_node, view, drug_kernel_dict,
                             prot_kernel_dict)

        # save simulation data resource tree to file.
        sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data_dict, transformer, flags, data_node, view, drug_kernel_dict, prot_kernel_dict):
    hyper_params = default_hparams_bopt(flags, *view)
    # Initialize the model and other related entities for training.
    if flags["cv"]:
        folds_data = []
        data_node.data = folds_data
        data_node.label = data_node.label + "cv"
        for k in range(flags["fold_num"]):
            k_node = DataNode(label="fold-%d" % k)
            folds_data.append(k_node)
            start_fold(k_node, data_dict, flags, hyper_params, tasks, trainer, transformer, view, drug_kernel_dict,
                       prot_kernel_dict, k)
    else:
        start_fold(data_node, data_dict, flags, hyper_params, tasks, trainer, transformer, view, drug_kernel_dict,
                   prot_kernel_dict)


def start_fold(sim_data_node, data_dict, flags, hyper_params, tasks, trainer, tranformer, view, drug_kernel_dict,
               prot_kernel_dict, k=None):
    data = trainer.data_provider(k, flags, data_dict)
    _data, reg_lambda, metrics = trainer.initialize(hparams=hyper_params,
                                                    train_dataset=data["train"],
                                                    val_dataset=data["val"],
                                                    test_dataset=data["test"],
                                                    drug_kernel_dict=drug_kernel_dict,
                                                    prot_kernel_dict=prot_kernel_dict)
    if flags["eval"]:
        pass
    else:
        # Train the model
        results = trainer.train(data, reg_lambda, metrics, tranformer, drug_kernel_dict, prot_kernel_dict,
                                sim_data_node=sim_data_node)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
            flags["cold_drug"] else "None"
        save_model(model, flags["model_dir"],
                   "{}_{}_{}_{}_{}_{:.4f}".format(flags["dataset"], '_'.join(view), flags["model_name"], split_label,
                                                  epoch, score))


def default_hparams_rand(flags, comp_view, prot_view):
    return {
        "reg_lambda": 0.1,
        "comp_view": comp_view,
        "prot_view": prot_view
    }


def default_hparams_bopt(flags, comp_view, prot_view):
    return {
        "reg_lambda": 0.1,
        "comp_view": comp_view,
        "prot_view": prot_view
    }


def get_hparam_config(flags, comp_view, prot_view):
    return {
        "reg_lambda": RealParam(min=0.1),
        "comp_view": ConstantParam(comp_view),
        "prot_view": ConstantParam(prot_view)
    }


class Flags(object):
    # enables using either object referencing or dict indexing to retrieve user passed arguments of flag objects.
    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        setattr(self, key, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kronecker Regularized Least Squares (Pahikkala et al., 2014")

    parser.add_argument("--dataset",
                        type=str,
                        default="davis",
                        help="Dataset name.")

    # Either CV or standard train-val(-test) split.
    scheme = parser.add_mutually_exclusive_group()
    scheme.add_argument("--fold_num",
                        default=-1,
                        type=int,
                        choices=range(3, 11),
                        help="Number of folds for cross-validation")
    scheme.add_argument("--test",
                        action="store_true",
                        help="Whether a test set should be included in the data split")

    parser.add_argument("--splitting_alg",
                        choices=["random", "scaffold", "butina", "index", "task"],
                        default="random",
                        type=str,
                        help="Data splitting algorithm to use.")
    parser.add_argument('--filter_threshold',
                        type=int,
                        default=6,
                        help='Threshold such that entities with observations no more than it would be filtered out.'
                        )
    parser.add_argument('--cold_drug',
                        default=False,
                        help='Flag of whether the split will leave "cold" drugs in the test data.',
                        action='store_true'
                        )
    parser.add_argument('--cold_target',
                        default=False,
                        help='Flag of whether the split will leave "cold" targets in the test data.',
                        action='store_true'
                        )
    parser.add_argument('--cold_drug_cluster',
                        default=False,
                        help='Flag of whether the split will leave "cold cluster" drugs in the test data.',
                        action='store_true'
                        )
    parser.add_argument('--predict_cold',
                        default=False,
                        help='Flag of whether the split will leave "cold" entities in the test data.',
                        action='store_true')
    parser.add_argument('--split_warm',
                        default=False,
                        help='Flag of whether the split will not leave "cold" entities in the test data.',
                        action='store_true'
                        )
    parser.add_argument('--model_dir',
                        type=str,
                        default='./model_dir',
                        help='Directory to store the log files in the training process.'
                        )
    parser.add_argument('--model_name',
                        type=str,
                        default='model-{}'.format(date_label),
                        help='Directory to store the log files in the training process.'
                        )
    parser.add_argument('--prot_desc_path',
                        action='append',
                        help='A list containing paths to protein descriptors.'
                        )
    parser.add_argument('--no_reload',
                        action="store_false",
                        dest='reload',
                        help='Whether datasets will be reloaded from existing ones or newly constructed.'
                        )
    parser.add_argument('--data_dir',
                        type=str,
                        default='../../data/',
                        help='Root folder of data (Davis, KIBA, Metz) folders.')
    parser.add_argument("--hparam_search",
                        action="store_true",
                        help="If true, hyperparameter searching would be performed.")
    parser.add_argument("--hparam_search_alg",
                        type=str,
                        default="bayopt_search",
                        help="Hyperparameter search algorithm to use. One of [bayopt_search, random_search]")
    parser.add_argument("--prot_view", "-pv",
                        type=str,
                        action="append",
                        help="The view to be simulated. One of [psc, rnn, pcnn]")
    parser.add_argument("--comp_view", "-cv",
                        type=str,
                        action="append",
                        help="The view to be simulated. One of [ecfp4, ecfp8, weave, gconv]")
    parser.add_argument("--eval",
                        action="store_true",
                        help="If true, a saved model is loaded and evaluated using CV")
    parser.add_argument("--eval_model_name",
                        default=None,
                        type=str,
                        help="The filename of the model to be loaded from the directory specified in --model_dir")

    args = parser.parse_args()

    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])
    setattr(flags, "cv", True if flags.fold_num > 2 else False)
    setattr(flags, "views", [(cv, pv) for cv, pv in zip(args.comp_view, args.prot_view)])
    main(flags)
