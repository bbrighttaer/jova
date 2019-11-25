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
import time
from collections import defaultdict
from datetime import datetime as dt
import torch
from torch.nn.functional import mse_loss
import numpy as np
from soek import RealParam
from soek.bopt import BayesianOptSearchCV
from soek.params import ConstantParam
from soek.rand import RandomSearchCV

import jova.metrics as mt
from jova.data import get_data, load_proteins
from jova.data.data import Pair
from jova.metrics import compute_model_performance
from jova.utils import Trainer, io
from jova.utils.io import save_model, save_dict_model
from jova.utils.sim_data import DataNode
from jova.utils.thread import UnboundedProgressbar

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

# seeds = [123, 124, 125]
seeds = [1, 8, 64]


class KronRLS(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset):
        data = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        # metrics
        metrics = [mt.Metric(mt.rms_score, np.nanmean),
                   mt.Metric(mt.concordance_index, np.nanmean),
                   mt.Metric(mt.pearson_r2_score, np.nanmean)]
        return data, hparams['reg_lambda'], metrics

    @staticmethod
    def data_provider(fold, flags, data):
        if not flags['cv']:
            print("Training scheme: train, validation" + (", test split" if flags['test'] else " split"))
            train_dataset = (data[1][0].X, data[1][0].y, data[1][0].w)
            valid_dataset = (data[1][1].X, data[1][1].y, data[1][1].w)
            test_dataset = None
            if flags['test']:
                test_dataset = (data[1][2].X, data[1][2].y, data[1][2].w)
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset}
        else:
            train_dataset = (data[1][fold][0].X, data[1][fold][0].y, data[1][fold][0].w)
            valid_dataset = (data[1][fold][1].X, data[1][fold][1].y, data[1][fold][1].w)
            test_dataset = (data[1][fold][2].X, data[1][fold][2].y, data[1][fold][2].w)
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset}
        return data

    @staticmethod
    def evaluate(eval_dict, y, y_pred, w, metrics, tasks, transformers):
        eval_dict.update(compute_model_performance(metrics, y_pred, y, w, transformers, tasks=tasks))
        # scoring
        rms = np.nanmean(eval_dict["nanmean-rms_score"])
        ci = np.nanmean(eval_dict["nanmean-concordance_index"])
        r2 = np.nanmean(eval_dict["nanmean-pearson_r2_score"])
        score = np.nanmean([ci, r2]) - rms
        return score

    @staticmethod
    def train(data, reg_lambda, metrics, transformer, drug_kernel_dict, prot_kernel_dict, tasks, sim_data_node,
              is_hsearch=False):
        start = time.time()
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        if sim_data_node:
            sim_data_node.data = [metrics_node]

        # Construct Kd and Kt
        train_mol = set()
        train_prot = set()
        labels = defaultdict(lambda: float)
        weights = defaultdict(lambda: float)
        for x, y, w in zip(*data['train']):
            mol, prot = x
            train_mol.add(mol)
            train_prot.add(prot)
            labels[Pair(mol, prot)] = float(y)
            weights[Pair(mol, prot)] = float(w)
        Kd = np.array([[drug_kernel_dict[Pair(c1, c2)] for c2 in train_mol] for c1 in train_mol], dtype=np.float)
        Kt = np.array([[prot_kernel_dict[Pair(p1, p2)] for p2 in train_prot] for p1 in train_prot], dtype=np.float)
        Y = np.array([[labels[Pair(c, p)] for p in train_prot] for c in train_mol], dtype=np.float)
        W = np.array([[weights[Pair(c, p)] for p in train_prot] for c in train_mol], dtype=np.float)
        assert (Y.shape == W.shape)
        Y = Y * W
        print('Kd.shape={}, Kt.shape={}, Y.shape={}, W.shape={}'.format(Kd.shape, Kt.shape, Y.shape, W.shape))

        # I'm impatient so I wanna see something :)
        pgbar = UnboundedProgressbar()
        pgbar.start()

        # Eigen decompositions
        Lambda, V = np.linalg.eigh(Kd)
        # Lambda, V = np.real(Lambda), np.real(V)
        Sigma, U = np.linalg.eigh(Kt)
        # Sigma, U = np.real(Sigma), np.real(U)

        # Compute C
        C = np.kron(np.diag(Lambda), np.diag(Sigma)) + reg_lambda * np.identity(Kd.shape[0] * Kt.shape[0])
        C = np.linalg.inv(C) @ np.ravel(U.T @ Y.T @ V)
        C = C.reshape(U.shape[0], V.shape[0])

        # compute weights
        A = U @ C @ V.T
        A = A.reshape(-1, 1)

        # assign weights
        entities_mat = np.array([[Pair(c, p) for p in train_prot] for c in train_mol], dtype=np.object).ravel()
        A_dict = {pair: a for pair, a in zip(entities_mat, A)}

        # Test / Validation
        eval_data = data['val']  # if is_hsearch else 'test']
        eval_mat = []
        y = eval_data[1]
        w = eval_data[2]
        for x_i in eval_data[0]:
            mol, prot = x_i
            row = [drug_kernel_dict[Pair(mol, pair.p1)] * prot_kernel_dict[Pair(prot, pair.p2)] for pair in A_dict]
            eval_mat.append(row)
        eval_mat = np.array(eval_mat, dtype=np.float)
        y_hat = eval_mat @ A
        assert (len(y_hat.shape) == len(w.shape))
        y_hat = y_hat * w
        loss = mse_loss(input=to_tensor(y_hat), target=to_tensor(y))

        # Alright, time to move on
        pgbar.stop()
        pgbar.join()

        # Metrics
        eval_dict = {}
        score = KronRLS.evaluate(eval_dict, y, y_hat, w, metrics, tasks, transformer)
        for m in eval_dict:
            if m in metrics_dict:
                metrics_dict[m].append(eval_dict[m])
            else:
                metrics_dict[m] = [eval_dict[m]]
        print('Evaluation loss={}, score={}, metrics={}'.format(loss.item(), score, str(eval_dict)))

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        return {'model': A_dict, 'score': score, 'epoch': 0}

    @staticmethod
    def evaluate_model():
        pass


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def to_tensor(array):
    return torch.from_numpy(array)


def main(flags):
    if len(flags["views"]) > 0:
        print("Single views for training: {}, num={}".format(flags["views"], len(flags["views"])))
    else:
        print("No views selected for training")

    for view in flags["views"]:
        cview, pview = view
        sim_label = "KronRLS_{}_{}".format(cview, pview)
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

            data_key = {"ecfp4": "KRLS_ECFP4",
                        "ecfp8": "KRLS_ECFP8"}.get(cview)
            data = get_data(data_key, flags, prot_sequences=prot_seq_dict, seed=seed)
            transformer = data[2]
            drug_kernel_dict, prot_kernel_dict = data[4]
            tasks = data[0]
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
                                   "data": data}
                n_iters = 3000
                extra_train_args = {"transformer": transformer,
                                    "drug_kernel_dict": drug_kernel_dict,
                                    "prot_kernel_dict": prot_kernel_dict,
                                    "tasks": tasks,
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
                                               save_model_fn=io.save_dict_model,
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
                invoke_train(trainer, tasks, data, transformer, flags, data_node, (cview, pview), drug_kernel_dict,
                             prot_kernel_dict)

        # save simulation data resource tree to file.
        sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data, transformer, flags, data_node, view, drug_kernel_dict, prot_kernel_dict):
    hyper_params = default_hparams_bopt(flags, *view)
    # Initialize the model and other related entities for training.
    if flags["cv"]:
        folds_data = []
        data_node.data = folds_data
        data_node.label = data_node.label + "cv"
        for k in range(flags["fold_num"]):
            k_node = DataNode(label="fold-%d" % k)
            folds_data.append(k_node)
            start_fold(k_node, data, flags, hyper_params, tasks, trainer, transformer, view, drug_kernel_dict,
                       prot_kernel_dict, k)
    else:
        start_fold(data_node, data, flags, hyper_params, tasks, trainer, transformer, view, drug_kernel_dict,
                   prot_kernel_dict)


def start_fold(sim_data_node, data, flags, hyper_params, tasks, trainer, tranformer, view, drug_kernel_dict,
               prot_kernel_dict, k=None):
    data = trainer.data_provider(k, flags, data)
    _data, reg_lambda, metrics = trainer.initialize(hparams=hyper_params,
                                                    train_dataset=data["train"],
                                                    val_dataset=data["val"],
                                                    test_dataset=data["test"])
    if flags["eval"]:
        pass
    else:
        # Train the model
        results = trainer.train(data, reg_lambda, metrics, tranformer, drug_kernel_dict, prot_kernel_dict,
                                tasks=tasks, sim_data_node=sim_data_node)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
            flags["cold_drug"] else "None"
        save_dict_model(model, flags["model_dir"],
                        "{}_{}_{}_{}_{}_{:.4f}".format(flags["dataset"], '_'.join(view), flags["model_name"],
                                                       split_label,
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
