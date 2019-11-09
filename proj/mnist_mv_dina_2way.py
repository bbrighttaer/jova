# Author: bbrighttaer
# Project: jova
# Date: 8/19/19
# Time: 9:51 AM
# File: mnist_mv.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import random
import time
from datetime import datetime as dt

import numpy as np
import sklearn.metrics as mt
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sch
import torch.utils.data as ds
# from torch.utils.tensorboard import SummaryWriter
from soek import RandomSearchCV, BayesianOptSearchCV, ConstantParam, RealParam, CategoricalParam, DiscreteParam, \
    LogRealParam
from tqdm import tqdm

from jova import cuda
from jova.data import Dataset
from jova.nn.models import TwoWayAttention, TwoWayForward
from jova.utils import Trainer
from jova.utils.io import save_model, load_model
from jova.utils.math import ExpAverage
from jova.utils.sim_data import DataNode
from jova.utils.train_helpers import load_data, split_mnist, trim_mnist, count_parameters, GradStats

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [123]


# writer = SummaryWriter(log_dir="~/tb_logs", comment="mnist_mv_%s" % date_label)


def get_out_shape(model, in_shape):
    return model(torch.zeros(1, *in_shape)).size()


class MnistPoc(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset):
        vws_lst = []
        l_dims = []
        for i in range(hparams["num_views"]):
            # create base model
            net = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4, padding=1),
                                nn.BatchNorm1d(8),
                                nn.ReLU(),
                                nn.Dropout(hparams["dprob"]),
                                nn.MaxPool1d(kernel_size=4, stride=1))

            # get the output shape of the base model using fake data trick
            base_shape = get_out_shape(net, hparams["views_in_shape"][i])
            l_dims.append(base_shape[-1])
            vws_lst.append(net)
        out_dim = sum(l_dims)
        model = nn.Sequential(TwoWayForward(*vws_lst),
                              TwoWayAttention(),
                              nn.Dropout(hparams["dprob"]),
                              nn.Linear(2 * 780, 10))

        if cuda:
            model = model.cuda()
        print("Number of trainable parameters = {}".format(count_parameters(model)))

        train_data_loader = ds.DataLoader(train_dataset, batch_size=hparams["tr_bsize"], shuffle=True)
        val_data_loader = ds.DataLoader(val_dataset, batch_size=hparams["val_bsize"], shuffle=False)

        # optimizer configuration
        optimizer = {
            "adadelta": torch.optim.Adadelta,
            "adagrad": torch.optim.Adagrad,
            "adam": torch.optim.Adam,
            "adamax": torch.optim.Adamax,
            "asgd": torch.optim.ASGD,
            "rmsprop": torch.optim.RMSprop,
            "Rprop": torch.optim.Rprop,
            "sgd": torch.optim.SGD,
        }.get(hparams["optimizer"].lower(), None)
        assert optimizer is not None, "{} optimizer could not be found"

        # filter optimizer arguments
        optim_kwargs = dict()
        optim_key = hparams["optimizer"]
        for k, v in hparams.items():
            if "optimizer__" in k:
                attribute_tup = k.split("__")
                if optim_key == attribute_tup[1] or attribute_tup[1] == "global":
                    optim_kwargs[attribute_tup[2]] = v
        optimizer = optimizer(model.parameters(), **optim_kwargs)

        # metrics
        metrics = (mt.accuracy_score,)
        return model, optimizer, {"train": train_data_loader,
                                  "val": val_data_loader}, metrics

    @staticmethod
    def data_provider(fold, views, tr_bsize=1, ts_bsize=1):
        num_split = 2
        (tr_v3_x, tr_v3_y), (tt_v3_x, tt_v3_y) = load_data('noisymnist_view1.gz',
                                                           'https://www2.cs.uic.edu/~vnoroozi/'
                                                           'noisy-mnist/noisymnist_view1.gz')
        (tr_v4_x, tr_v4_y), (tt_v4_x, tt_v4_y) = load_data('noisymnist_view2.gz',
                                                           'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/'
                                                           'noisymnist_view2.gz')

        (tr_v1_x, tr_v1_y), (tr_v2_x, tr_v2_y) = split_mnist(tr_v3_x, tr_v3_y, num_split)
        (tt_v1_x, tt_v1_y), (tt_v2_x, tt_v2_y) = split_mnist(tt_v3_x, tt_v3_y, num_split)

        tr_lst, tt_lst = [], []

        for v in views:
            if v == "v1":
                tr_lst.append((tr_v1_x, tr_v1_y))
                tt_lst.append((tt_v1_x, tt_v1_y))
            if v == "v2":
                tr_lst.append((tr_v2_x, tr_v2_y))
                tt_lst.append((tt_v2_x, tt_v2_y))
            if v == "v3":
                tr_lst.append((tr_v3_x, tr_v3_y))
                tt_lst.append((tt_v3_x, tt_v3_y))
            if v == "v4":
                tr_lst.append((tr_v4_x, tr_v4_y))
                tt_lst.append((tt_v4_x, tt_v4_y))

        train_data = trim_mnist(tr_lst, batch_size=tr_bsize)
        test_data = trim_mnist(tt_lst, batch_size=ts_bsize)

        return {"train": Dataset(train_data),
                "val": Dataset(test_data)}

    @staticmethod
    def evaluate(eval_dict, y, y_pred, metrics):
        for metric in metrics:
            y_pred = torch.max(y_pred, dim=1)[1].cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            eval_dict[metric.__name__] = metric(y, y_pred)
        return np.mean(list(eval_dict.values()))

    @staticmethod
    def train(eval_fn, model, optimizer, data_loaders, metrics, n_iters=5000, sim_data_node=None, epoch_ckpt=(2, 1.5)):
        start = time.time()
        best_model_wts = model.state_dict()
        best_score = -10000
        best_epoch = -1
        terminate_training = False
        n_epochs = n_iters // len(data_loaders["train"])
        scheduler = sch.StepLR(optimizer, step_size=30, gamma=0.01)
        criterion = nn.CrossEntropyLoss()
        e_avg = ExpAverage(.9)
        grad_stats = GradStats(model, beta=0.)

        # sub-nodes of sim data resource
        loss_lst = []
        train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [train_loss_node, metrics_node, scores_node]

        try:
            # Main training loop
            for epoch in range(n_epochs):
                if terminate_training:
                    print("Terminating training...")
                    break
                for phase in ["train", "val"]:
                    if phase == "train":
                        print("Training....")
                        # Training mode
                        model.train()
                    else:
                        print("Validation...")
                        # Evaluation mode
                        model.eval()
                        # model.plot_images()

                    data_size = 0.
                    epoch_losses = []
                    epoch_scores = []

                    # Iterate through mini-batches
                    i = 0
                    for batch in tqdm(data_loaders[phase]):
                        Xs = [v[0].unsqueeze(dim=1).cuda() if cuda else v[0].unsqueeze(dim=1) for v in batch]
                        Ys = [v[1].cuda() if cuda else v[1] for v in batch]

                        optimizer.zero_grad()

                        # forward propagation
                        # track history if only in train
                        with torch.set_grad_enabled(phase == "train"):
                            y_pred = model(Xs)
                            y = Ys[0].squeeze()
                            loss = criterion(y_pred, y)

                        if phase == "train":
                            # optimization ops
                            loss = loss
                            loss.backward()

                            # for epoch stats
                            epoch_losses.append(loss.item())

                            # for sim data resource
                            loss_lst.append(loss.item())
                            print("\tEpoch={}/{}, batch={}/{}, loss={:.4f} ".format(epoch + 1, n_epochs,
                                                                                    i + 1,
                                                                                    len(data_loaders[phase]),
                                                                                    loss.item()
                                                                                    # grad_stats.stats()
                                                                                    ))
                            optimizer.step()
                        else:
                            if str(loss.item()) != "nan":  # useful in hyperparameter search
                                eval_dict = {}
                                score = eval_fn(eval_dict, y, y_pred, metrics)
                                # for epoch stats
                                epoch_scores.append(score)

                                # for sim data resource
                                scores_lst.append(score)
                                for m in eval_dict:
                                    if m in metrics_dict:
                                        metrics_dict[m].append(eval_dict[m])
                                    else:
                                        metrics_dict[m] = [eval_dict[m]]

                                print("\nEpoch={}/{}, batch={}/{}, "
                                      "evaluation results= {}, score={}".format(epoch + 1, n_epochs, i + 1,
                                                                                len(data_loaders[phase]),
                                                                                eval_dict, score))
                            else:
                                terminate_training = True

                        i += 1
                    # End of mini=batch iterations.

                    if phase == "train":
                        # Adjust the learning rate.
                        scheduler.step()

                        ep_loss = np.nanmean(epoch_losses)
                        e_avg.update(ep_loss)
                        if epoch % epoch_ckpt[0] - 1 == 0 and epoch > 0:
                            if e_avg.value > epoch_ckpt[1]:
                                terminate_training = True
                        print("\nPhase: {}, avg task loss={:.4f}, ".format(phase, ep_loss))

                    else:
                        mean_score = np.mean(epoch_scores)
                        if best_score < mean_score:
                            best_score = mean_score
                            best_model_wts = copy.deepcopy(model.state_dict())
                            best_epoch = epoch
        except RuntimeError as e:
            print(str(e))

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        try:
            model.load_state_dict(best_model_wts)
        except RuntimeError as e:
            print(str(e))
        print("Number of trainable parameters = {}".format(count_parameters(model)))
        return model, best_score, best_epoch

    @staticmethod
    def evaluate_model(eval_fn, model, model_dir, model_name, data_loaders, metrics, sim_data_node=None):
        # Load the saved model and put it in evaluation mode
        model.load_state_dict(load_model(model_dir, model_name))
        model.eval()
        model.plot_representations()

        print("Model evaluation...")
        start = time.time()
        n_epochs = 1
        phase = "val"
        criterion = nn.CrossEntropyLoss()

        # sub-nodes of sim data resource
        loss_lst = []
        train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [train_loss_node, metrics_node, scores_node]

        # Main training loop
        for epoch in range(n_epochs):
            epoch_losses = []
            epoch_scores = []

            # Iterate through mini-batches
            i = 0
            Xs = Ys = None
            for batch in tqdm(data_loaders[phase]):
                Xs = [v[0].unsqueeze(dim=1).cuda() if cuda else v[0].unsqueeze(dim=1) for v in batch]
                Ys = [v[1].cuda() if cuda else v[1] for v in batch]

                # forward propagation
                # track history if only in train
                with torch.set_grad_enabled(False):
                    rank_loss, outputs = model(*Xs)
                    pred_loss = 0.
                    for y_pred, y in zip(outputs, Ys):
                        pred_loss = pred_loss + criterion(y_pred, y.squeeze())
                    pred_loss = pred_loss / float(len(Xs))

                    # optimization ops
                    loss = pred_loss + rank_loss

                    # for epoch stats
                    epoch_losses.append(loss.item())

                    # for sim data resource
                    loss_lst.append(loss.item())
                    print("\tEpoch={}/{}, batch={}/{}, pred_loss={:.4f}, "
                          "rank_loss={:.4f}, ".format(epoch + 1, n_epochs,
                                                      i + 1,
                                                      len(data_loaders[phase]),
                                                      pred_loss.item(),
                                                      rank_loss,
                                                      # grad_stats.stats()
                                                      ))

                    eval_dict = {}
                    score = eval_fn(eval_dict, Ys, outputs, metrics)
                    # for epoch stats
                    epoch_scores.append(score)

                    # for sim data resource
                    scores_lst.append(score)
                    for m in eval_dict:
                        if m in metrics_dict:
                            metrics_dict[m].append(eval_dict[m])
                        else:
                            metrics_dict[m] = [eval_dict[m]]

                    print("\nEpoch={}/{}, batch={}/{}, "
                          "evaluation results= {}, score={}".format(epoch + 1, n_epochs, i + 1,
                                                                    len(data_loaders[phase]),
                                                                    eval_dict, score))

                i += 1
            # End of mini=batch iterations.

        duration = time.time() - start
        print('\nModel evaluation duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))


def main(flags):
    sim_label = "CUDA={}, views={}, coord={}".format(cuda, flags.views, flags.coord)
    print(sim_label)

    # Simulation data resource tree
    sim_data = DataNode(label=sim_label)
    nodes_list = []
    sim_data.data = nodes_list

    for seed in seeds:
        # for data collection of this round of simulation.
        data_node = DataNode(label="seed_%d" % seed)
        nodes_list.append(data_node)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        trainer = MnistPoc()

        k = 1

        if flags.hparam_search:
            print("Hyperparameter search enabled: {}".format(flags.hparam_search_alg))

            # arguments to callables
            extra_init_args = {}
            extra_data_args = {"views": flags.views}
            extra_train_args = {"n_iters": 5000}

            hparams_conf = get_hparam_config(flags)

            search_alg = {"random_search": RandomSearchCV,
                          "bayopt_search": BayesianOptSearchCV}.get(flags.hparam_search_alg, BayesianOptSearchCV)

            hparam_search = search_alg(hparam_config=hparams_conf,
                                       num_folds=k,
                                       initializer=trainer.initialize,
                                       data_provider=trainer.data_provider,
                                       train_fn=trainer.train,
                                       eval_fn=trainer.evaluate,
                                       save_model_fn=save_model,
                                       init_args=extra_init_args,
                                       data_args=extra_data_args,
                                       train_args=extra_train_args,
                                       data_node=data_node,
                                       split_label="train_val",
                                       sim_label=sim_label,
                                       dataset_label="mnist",
                                       results_file="{}_{}_poc_{}.csv".format(flags.hparam_search_alg, sim_label,
                                                                              date_label))
            stats = hparam_search.fit(model_dir="param_search_models",
                                      model_name=flags.model_name, max_iter=40, seed=seed)
            print(stats)
            print("Best params = {}".format(stats.best(m="max")))
        else:
            data = trainer.data_provider(fold=k, views=flags.views)
            model, optimizer, data_loaders, metrics = trainer.initialize(hparams=get_hparams(flags),
                                                                         train_dataset=data["train"],
                                                                         val_dataset=data["val"])
            if flags.eval:
                trainer.evaluate_model(trainer.evaluate, model, flags.model_dir, flags.eval_model_name, data_loaders,
                                       metrics, data_node)
            else:
                # Train the model
                model, score, epoch = trainer.train(trainer.evaluate, model, optimizer, data_loaders, metrics,
                                                    n_iters=10000, sim_data_node=data_node)
                # Save the model
                save_model(model, flags.model_dir,
                           "mnist_{}_{}_poc_{}_{:.5f}".format(sim_label, flags.model_name, epoch, score))

    # save simulation data resource tree to file.
    sim_data.to_json(path="./analysis/")

    # TB
    # writer.close()


def get_hparams(flags):
    hparams = {
        "coord": flags.coord,
        "latent_dim": 50,
        "num_views": len(flags.views),
        "attn_heads": 2,
        "views_in_shape": [(1, 392), (1, 392), (1, 784), (1, 784)],
        "tr_bsize": 128,
        "val_bsize": 128,
        "dina_out_dim": 512,
        "proj_pool_func": "max",

        "dprob": 0.25,

        # optimizer params
        "optimizer": "rmsprop",
        "optimizer__global__weight_decay": 0.0009111,
        "optimizer__global__lr": 0.00025737
    }
    return hparams


def get_hparam_config(flags):
    return {
        "num_views": ConstantParam(len(flags.views)),
        "views_in_shape": ConstantParam([(1, 392), (1, 392), (1, 784), (1, 784)]),
        "tr_bsize": CategoricalParam(choices=[64, 128, 256, 512]),
        "val_bsize": ConstantParam(128),
        "dprob": RealParam(min=0.1, max=0.5),
        "dina_out_dim": CategoricalParam(choices=[64, 128, 256, 512]),
        "attn_heads": DiscreteParam(min=1, max=6),
        "proj_pool_func": CategoricalParam(choices=['avg', 'max']),

        # optimizer params
        "optimizer": CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
        "optimizer__global__weight_decay": LogRealParam(),
        "optimizer__global__lr": LogRealParam(),

        # # SGD
        "optimizer__sgd__nesterov": CategoricalParam(choices=[True, False]),
        "optimizer__sgd__momentum": LogRealParam(),
        # "optimizer__sgd__lr": LogRealParam(),

        # ADAM
        # "optimizer__adam__lr": LogRealParam(),
        "optimizer__adam__amsgrad": CategoricalParam(choices=[True, False]),

        # Adadelta
        # "optimizer__adadelta__lr": LogRealParam(),
        # "optimizer__adadelta__weight_decay": LogRealParam(),
        "optimizer__adadelta__rho": LogRealParam(),

        # Adagrad
        # "optimizer__adagrad__lr": LogRealParam(),
        # "optimizer__adagrad__lr_decay": LogRealParam(),
        # "optimizer__adagrad__weight_decay": LogRealParam(),

        # Adamax
        # "optimizer__adamax__lr": LogRealParam(),
        # "optimizer__adamax__weight_decay": LogRealParam(),

        # RMSprop
        # "optimizer__rmsprop__lr": LogRealParam(),
        # "optimizer__rmsprop__weight_decay": LogRealParam(),
        "optimizer__rmsprop__momentum": LogRealParam(),
        "optimizer__rmsprop__centered": CategoricalParam(choices=[True, False])
    }


class Flags(object):
    # enables using either object referencing or dict indexing to retrieve user passed arguments of flag objects.
    def __getitem__(self, item):
        return self.__dict__[item]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparam_search",
                        action="store_true",
                        help="If true, hyperparameter searching would be performed.")
    parser.add_argument("--hparam_search_alg",
                        type=str,
                        default="bayopt_search",
                        help="Hyperparameter search algorithm to use. One of [bayopt_search, random_search]")
    parser.add_argument("--view",
                        action="append",
                        dest="views",
                        help="The view to be simulated. One of [v1, v2, v3, v4]")
    parser.add_argument('--model_dir',
                        type=str,
                        default='./model_dir',
                        help='Directory of model'
                        )
    parser.add_argument('--model_name',
                        type=str,
                        default='model-{}'.format(date_label),
                        help='Name of model'
                        )
    parser.add_argument("--eval",
                        action="store_true",
                        help="If true, a saved model is loaded and evaluated")
    parser.add_argument("--eval_model_name",
                        default=None,
                        type=str,
                        help="The filename of the model to be loaded from the directory specified in --model_dir")
    parser.add_argument("--no_mv",
                        action="store_false",
                        dest="coord",
                        help="Controls whether coordinated multi-view is active or not")

    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])

    main(flags)
