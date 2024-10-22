# Author: bbrighttaer
# Project: jova
# Date: 7/2/19
# Time: 1:24 PM
# File: singleview.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import json
import os
import random
import time
from datetime import datetime as dt

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim.lr_scheduler as sch
from soek import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import jova.metrics as mt
import jova.utils.io
from jova import cuda
from jova.data import batch_collator, get_data, load_proteins, DtiDataset
from jova.metrics import compute_model_performance
from jova.nn.layers import GraphConvLayer, GraphPool, GraphGather, LambdaLayer, Reshape
from jova.nn.models import create_fcn_layers, WeaveModel, GraphConvSequential, PairSequential, GraphNeuralNet, Prot2Vec, \
    ProteinRNN, ProteinCNNAttention, ProtCnnForward
from jova.trans import undo_transforms
from jova.utils import Trainer
from jova.utils.args import FcnArgs, WeaveLayerArgs, WeaveGatherArgs, Flags
from jova.utils.io import save_model, load_model, load_pickle
from jova.utils.math import ExpAverage, Count
from jova.utils.tb import TBMeanTracker
from jova.utils.train_helpers import count_parameters, parse_hparams

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1, 8, 64]
torch.cuda.set_device(0)


def create_prot_net(hparams, protein_profile):
    valid_opts = ["rnn", "psc", "pcnna", "p2v"]  # pccna - PCNN with Attention
    model_type = hparams["prot"]["model_type"]
    assert (model_type in valid_opts), "Valid protein types: {}".format(str(valid_opts))
    prot_dim = hparams["prot"]["dim"]
    window = hparams["prot"]["window"]
    if model_type == "rnn":
        model = nn.Sequential(Prot2Vec(protein_profile,
                                       vocab_size=hparams["prot"]["vocab_size"],
                                       embedding_dim=prot_dim,
                                       batch_first=True),
                              ProteinRNN(in_dim=prot_dim * window,
                                         hidden_dim=prot_dim,
                                         dropout=hparams["dprob"],
                                         batch_first=True),
                              LambdaLayer(lambda x: torch.sum(x, dim=1, keepdim=True)),
                              Reshape(shape=(-1, prot_dim)))
    elif model_type == "p2v":
        model = nn.Sequential(Prot2Vec(protein_profile=protein_profile,
                                       vocab_size=hparams["prot"]["vocab_size"],
                                       embedding_dim=prot_dim,
                                       batch_first=True),
                              LambdaLayer(lambda x: torch.sum(x, dim=1, keepdim=True)),
                              Reshape(shape=(-1, prot_dim * window)),
                              nn.Linear(prot_dim * window, prot_dim),
                              nn.BatchNorm1d(prot_dim),
                              nn.ReLU(),
                              nn.Dropout())
    elif model_type == "pcnna":
        model = (Prot2Vec(protein_profile=protein_profile,
                          vocab_size=hparams["prot"]["vocab_size"],
                          embedding_dim=prot_dim,
                          batch_first=True),
                 ProteinCNNAttention(dim=prot_dim,
                                     window=window,
                                     activation="relu",
                                     num_layers=hparams["prot"]["pcnn_num_layers"]))
    else:  # psc
        model = nn.Sequential(nn.Identity())
    return model


def create_ecfp_net(hparams):
    return nn.Identity(), (hparams["ecfp8"]["dim"], hparams["prot"]["dim"])


def create_weave_net(hparams):
    weave_args = (
        WeaveLayerArgs(n_atom_input_feat=75,
                       n_pair_input_feat=14,
                       n_atom_output_feat=50,
                       n_pair_output_feat=50,
                       n_hidden_AA=50,
                       n_hidden_PA=50,
                       n_hidden_AP=50,
                       n_hidden_PP=50,
                       update_pair=True,
                       activation='relu',
                       batch_norm=True,
                       dropout=hparams["dprob"]
                       ),
        WeaveLayerArgs(n_atom_input_feat=50,
                       n_pair_input_feat=50,
                       n_atom_output_feat=50,
                       n_pair_output_feat=50,
                       n_hidden_AA=50,
                       n_hidden_PA=50,
                       n_hidden_AP=50,
                       n_hidden_PP=50,
                       update_pair=True,
                       batch_norm=True,
                       dropout=hparams["dprob"],
                       activation='relu'),
    )
    wg_args = WeaveGatherArgs(conv_out_depth=50, gaussian_expand=True, n_depth=hparams["weave"]["dim"])
    weave_model = WeaveModel(weave_args, wg_args)
    civ_dim = (hparams["weave"]["dim"], hparams["prot"]["dim"])
    return weave_model, civ_dim


def create_gconv_net(hparams):
    gconv_model = GraphConvSequential(GraphConvLayer(in_dim=75, out_dim=64),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      GraphPool(),

                                      GraphConvLayer(in_dim=64, out_dim=64),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      GraphPool(),

                                      nn.Linear(in_features=64, out_features=hparams["gconv"]["dim"]),
                                      nn.BatchNorm1d(hparams["gconv"]["dim"]),
                                      nn.ReLU(),
                                      nn.Dropout(hparams["dprob"]),
                                      GraphGather())
    civ_dim = (hparams["gconv"]["dim"] * 2, hparams["prot"]["dim"])
    return gconv_model, civ_dim


def create_gnn_net(hparams):
    dim = hparams["gnn"]["dim"]
    gnn_model = GraphNeuralNet(num_fingerprints=hparams["gnn"]["fingerprint_size"], embedding_dim=dim,
                               num_layers=hparams["gnn"]["num_layers"])
    civ_dim = (dim, hparams["prot"]["dim"])
    return gnn_model, civ_dim


def create_feedforwardnet(civ_dim, hparams, compound_model, protein_profile):
    if hparams["prot"]["model_type"] == "pcnna":
        p = 2 * hparams["prot"]["dim"]
        base_model = ProtCnnForward(*create_prot_net(hparams, protein_profile),
                                    nn.Sequential(compound_model, nn.Linear(*civ_dim)))
    else:
        base_model = PairSequential(mod1=(compound_model,), mod2=(create_prot_net(hparams, protein_profile),))
        p = np.sum(civ_dim)

    fcn_args = []
    fcn_layers = hparams["hdims"]
    if not isinstance(fcn_layers, list):
        fcn_layers = [fcn_layers]
    for dim in fcn_layers:
        conf = FcnArgs(in_features=p,
                       out_features=dim,
                       activation='relu',
                       batch_norm=True,
                       dropout=hparams["dprob"])
        fcn_args.append(conf)
        p = dim
    fcn_args.append(FcnArgs(in_features=p, out_features=hparams["output_dim"]))
    fcn_layers = create_fcn_layers(fcn_args)
    model = nn.Sequential(base_model, *fcn_layers)
    return model


class SingleViewDTI(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset, protein_profile, cuda_devices=None,
                   mode="regression"):

        # create network
        create_func = {"ecfp4": create_ecfp_net,
                       "ecfp8": create_ecfp_net,
                       "weave": create_weave_net,
                       "gconv": create_gconv_net,
                       "gnn": create_gnn_net}.get(hparams["comp_view"])
        comp_model, civ_dim = create_func(hparams)
        model = create_feedforwardnet(civ_dim, hparams, comp_model, protein_profile)
        print("Number of trainable parameters = {}".format(count_parameters(model)))
        try:
            if cuda:
                model = model.cuda()
        except RuntimeError as e:
            print(str(e))

        # data loaders
        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=hparams["tr_batch_size"],
                                       shuffle=True,
                                       collate_fn=lambda x: x)
        val_data_loader = DataLoader(dataset=val_dataset,
                                     batch_size=hparams["val_batch_size"],
                                     shuffle=False,
                                     collate_fn=lambda x: x)
        test_data_loader = None
        if test_dataset is not None:
            test_data_loader = DataLoader(dataset=test_dataset,
                                          batch_size=hparams["test_batch_size"],
                                          shuffle=False,
                                          collate_fn=lambda x: x)

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
        metrics = [mt.Metric(mt.rms_score, np.nanmean),
                   mt.Metric(mt.concordance_index, np.nanmean),
                   mt.Metric(mt.pearson_r2_score, np.nanmean)]
        return model, optimizer, {"train": train_data_loader,
                                  "val": val_data_loader,
                                  "test": test_data_loader}, metrics

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
    def train(model, optimizer, data_loaders, metrics, transformers_dict, prot_desc_dict, tasks, view,
              n_iters=5000, is_hsearch=False, sim_data_node=None, epoch_ckpt=(2, 1.0), tb_writer=None):
        tb_writer = tb_writer()
        comp_view, prot_view = view
        start = time.time()
        best_model_wts = model.state_dict()
        best_score = -10000
        best_epoch = -1
        terminate_training = False
        e_avg = ExpAverage(.01)
        n_epochs = n_iters // len(data_loaders["train"])
        scheduler = sch.StepLR(optimizer, step_size=400, gamma=0.01)
        criterion = torch.nn.MSELoss()

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
            tb_idx = {'train': Count(), 'val': Count(), 'test': Count()}
            for epoch in range(n_epochs):
                if terminate_training:
                    print("Terminating training...")
                    break
                for phase in ["train", "val" if is_hsearch else "test"]:
                    if phase == "train":
                        print("Training....")
                        # Training mode
                        model.train()
                    else:
                        print("Validation...")
                        # Evaluation mode
                        model.eval()

                    data_size = 0.
                    epoch_losses = []
                    epoch_scores = []

                    # Iterate through mini-batches
                    i = 0
                    with TBMeanTracker(tb_writer, 10) as tracker:
                        for batch in tqdm(data_loaders[phase]):
                            batch_size, data = batch_collator(batch, prot_desc_dict, spec=comp_view)
                            # Data
                            if prot_view in ["p2v", "rnn", "pcnn", "pcnna"]:
                                protein_x = data[comp_view][0][2]
                            else:  # then it's psc
                                protein_x = data[comp_view][0][1]
                            if comp_view == "gconv":
                                # graph data structure is: [(compound data, batch_size), protein_data]
                                X = ((data[comp_view][0][0], batch_size), protein_x)
                            else:
                                X = (data[comp_view][0][0], protein_x)
                            y = data[comp_view][1]
                            w = data[comp_view][2]
                            y = np.array([k for k in y], dtype=np.float)
                            w = np.array([k for k in w], dtype=np.float)

                            optimizer.zero_grad()

                            # forward propagation
                            # track history if only in train
                            with torch.set_grad_enabled(phase == "train"):
                                outputs = model(X)
                                target = torch.from_numpy(y).float()
                                weights = torch.from_numpy(w).float()
                                if cuda:
                                    target = target.cuda()
                                    weights = weights.cuda()
                                outputs = outputs * weights
                                target = target * weights
                                loss = criterion(outputs, target)

                            if str(loss.item()) == "nan":
                                terminate_training = True
                                break

                            # metrics
                            eval_dict = {}
                            score = SingleViewDTI.evaluate(eval_dict, y, outputs, w, metrics, tasks,
                                                           transformers_dict[comp_view])

                            # TBoard info
                            tracker.track("%s/loss" % phase, loss.item(), tb_idx[phase].IncAndGet())
                            tracker.track("%s/score" % phase, score, tb_idx[phase].i)
                            for k in eval_dict:
                                tracker.track('{}/{}'.format(phase, k), eval_dict[k], tb_idx[phase].i)

                            if phase == "train":
                                print("\tEpoch={}/{}, batch={}/{}, loss={:.4f}".format(epoch + 1, n_epochs, i + 1,
                                                                                       len(data_loaders[phase]),
                                                                                       loss.item()))
                                # for epoch stats
                                epoch_losses.append(loss.item())

                                # for sim data resource
                                loss_lst.append(loss.item())

                                # optimization ops
                                loss.backward()
                                optimizer.step()
                            else:
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
                            data_size += batch_size
                    # End of mini=batch iterations.
                    if phase == "train":
                        ep_loss = np.nanmean(epoch_losses)
                        e_avg.update(ep_loss)
                        if epoch % (epoch_ckpt[0] - 1) == 0 and epoch > 0:
                            if e_avg.value > epoch_ckpt[1]:
                                terminate_training = True

                        # Adjust the learning rate.
                        scheduler.step()
                        print("\nPhase: {}, avg task loss={:.4f}, ".format(phase, np.nanmean(epoch_losses)))
                    else:
                        mean_score = np.mean(epoch_scores)
                        if best_score < mean_score:
                            best_score = mean_score
                            best_model_wts = copy.deepcopy(model.state_dict())
                            best_epoch = epoch
        except Exception as e:
            print(str(e))
        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        model.load_state_dict(best_model_wts)
        return {'model': model, 'score': best_score, 'epoch': best_epoch}

    @staticmethod
    def evaluate_model(model, model_dir, model_name, data_loaders, metrics, transformers_dict, prot_desc_dict,
                       tasks, view, sim_data_node=None):
        comp_view, prot_view = view
        # load saved model and put in evaluation mode
        model.load_state_dict(load_model(model_dir, model_name, dvc=torch.device('cuda:0')))
        model.eval()

        print("Model evaluation...")
        start = time.time()
        n_epochs = 1

        # sub-nodes of sim data resource
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)
        predicted_vals = []
        true_vals = []
        model_preds_node = DataNode(label="model_predictions", data={"y": true_vals,
                                                                     "y_pred": predicted_vals})

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [metrics_node, scores_node, model_preds_node]

        # Main evaluation loop
        for epoch in range(n_epochs):

            for phase in ["test"]:
                # Iterate through mini-batches
                i = 0
                for batch in tqdm(data_loaders[phase]):
                    batch_size, data = batch_collator(batch, prot_desc_dict, spec=comp_view)
                    # Data
                    if prot_view in ["p2v", "rnn", "pcnn", "pcnna"]:
                        protein_x = data[comp_view][0][2]
                    else:  # then it's psc
                        protein_x = data[comp_view][0][1]
                    if comp_view == "gconv":
                        # graph data structure is: [(compound data, batch_size), protein_data]
                        X = ((data[comp_view][0][0], batch_size), protein_x)
                    else:
                        X = (data[comp_view][0][0], protein_x)
                    y = data[comp_view][1]
                    w = data[comp_view][2]
                    y = np.array([k for k in y], dtype=np.float)
                    w = np.array([k for k in w], dtype=np.float)

                    # prediction
                    y_predicted = model(X)

                    # apply transformers
                    predicted_vals.extend(undo_transforms(y_predicted.cpu().detach().numpy(),
                                                          transformers_dict[comp_view]).squeeze().tolist())
                    true_vals.extend(
                        undo_transforms(y, transformers_dict[comp_view]).astype(np.float).squeeze().tolist())

                    eval_dict = {}
                    score = SingleViewDTI.evaluate(eval_dict, y, y_predicted, w, metrics, tasks,
                                                   transformers_dict[comp_view])

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


def main(id, flags):
    if len(flags["views"]) > 0:
        print("Single views for training: {}, num={}".format(flags["views"], len(flags["views"])))
    else:
        print("No views selected for training")

    for view in flags["views"]:
        split_label = flags.split
        dataset_lbl = flags["dataset_name"]
        mode = "eval" if flags["eval"] else "train"
        if flags.cv:
            mode += 'cv'
        cview, pview = view
        sim_label = f"{dataset_lbl}_{split_label}_single_view_{cview}_{pview}_{mode}"
        print("CUDA={}, {}".format(cuda, sim_label))

        # Simulation data resource tree
        # node_label = "{}_{}_{}_{}_{}_{}".format(dataset_lbl, cview, pview, split_label, mode, date_label)
        node_label = json.dumps({'model_family': 'singleview',
                                 'dataset': dataset_lbl,
                                 'cview': cview,
                                 'pview': pview,
                                 'split': split_label,
                                 'mode': mode,
                                 'seeds': '-'.join([str(s) for s in seeds]),
                                 'date': date_label})
        sim_data = DataNode(label=node_label)
        nodes_list = []
        sim_data.data = nodes_list

        num_cuda_dvcs = torch.cuda.device_count()
        cuda_devices = None if num_cuda_dvcs == 1 else [i for i in range(1, num_cuda_dvcs)]

        prot_desc_dict, prot_seq_dict = load_proteins(flags['prot_desc_path'])
        prot_profile = load_pickle(file_name=flags['prot_profile'])
        prot_vocab = load_pickle(file_name=flags['prot_vocab'])
        flags["prot_vocab_size"] = len(prot_vocab)

        # For searching over multiple seeds
        hparam_search = None

        for seed in seeds:
            summary_writer_creator = lambda: SummaryWriter(
                log_dir="tb_singles_hs/{}_{}_{}/".format(sim_label, seed, dt.now().strftime("%Y_%m_%d__%H_%M_%S")))

            # for data collection of this round of simulation.
            data_node = DataNode(label="seed_%d" % seed)
            nodes_list.append(data_node)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # load data
            print('-------------------------------------')
            print('Running on dataset: %s' % dataset_lbl)
            print('-------------------------------------')

            data_dict = dict()
            transformers_dict = dict()
            data_key = {"ecfp4": "ECFP4",
                        "ecfp8": "ECFP8",
                        "weave": "Weave",
                        "gconv": "GraphConv",
                        "gnn": "GNN"}.get(cview)
            data_dict[cview] = get_data(data_key, flags, prot_sequences=prot_seq_dict, seed=seed)
            transformers_dict[cview] = data_dict[cview][2]
            flags["gnn_fingerprint"] = data_dict[cview][3]

            tasks = data_dict[cview][0]
            # multi-task or single task is determined by the number of tasks w.r.t. the dataset loaded
            flags["tasks"] = tasks

            trainer = SingleViewDTI()

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
                extra_init_args = {"mode": "regression",
                                   "cuda_devices": cuda_devices,
                                   "protein_profile": prot_profile}
                extra_data_args = {"flags": flags,
                                   "data_dict": data_dict}
                n_iters = 3000
                extra_train_args = {"transformers_dict": transformers_dict,
                                    "prot_desc_dict": prot_desc_dict,
                                    "tasks": tasks,
                                    "n_iters": n_iters,
                                    "is_hsearch": True,
                                    "view": view,
                                    "tb_writer": summary_writer_creator}

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
                                               save_model_fn=jova.utils.io.save_model,
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
                print("Best params = {}".format(stats.best()))
            else:
                invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node, view,
                             prot_profile, summary_writer_creator)

        # save simulation data resource tree to file.
        sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node, view,
                 prot_profile, summary_writer_creator):
    try:
        hfile = os.path.join('soek_res', get_hparam_file(*view))
        exists = os.path.exists(hfile)
        status = 'Found' if exists else 'Not Found, switching to default hyperparameters'
        print(f'Hyperparameters file:{hfile}, status={status}')
        if not exists:
            raise FileNotFoundError(f'{hfile} not found')
        hyper_params = parse_hparams(hfile)
        hyper_params['gnn']['fingerprint_size'] = len(flags["gnn_fingerprint"]) \
            if flags["gnn_fingerprint"] is not None else 0
        hyper_params['output_dim'] = len(tasks)
    except:
        hyper_params = default_hparams_bopt(flags, *view)

    # Initialize the model and other related entities for training.
    if flags["cv"]:
        folds_data = []
        data_node.data = folds_data
        data_node.label = data_node.label + "cv"
        for k in range(flags["fold_num"]):
            k_node = DataNode(label="fold-%d" % k)
            folds_data.append(k_node)
            start_fold(k_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
                       transformers_dict, view, prot_profile, summary_writer_creator, k)
    else:
        start_fold(data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
                   transformers_dict, view, prot_profile, summary_writer_creator)


def start_fold(sim_data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
               transformers_dict, view, protein_profile, summary_writer_creator=None, k=None):
    data = trainer.data_provider(k, flags, data_dict)
    model, optimizer, data_loaders, metrics = trainer.initialize(hparams=hyper_params,
                                                                 train_dataset=data["train"],
                                                                 val_dataset=data["val"],
                                                                 test_dataset=data["test"],
                                                                 protein_profile=protein_profile)
    if flags["eval"]:
        trainer.evaluate_model(model, flags["model_dir"], flags["eval_model_name"],
                               data_loaders, metrics, transformers_dict,
                               prot_desc_dict, tasks, view=view, sim_data_node=sim_data_node)
    else:
        # Train the model
        results = trainer.train(model, optimizer, data_loaders, metrics, transformers_dict, prot_desc_dict,
                                tasks, n_iters=10000, view=view, sim_data_node=sim_data_node,
                                tb_writer=summary_writer_creator)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        save_model(model, flags["model_dir"],
                   "{}_{}_{}_{}_{}_{:.4f}".format(flags["dataset_name"], '_'.join(view), flags["model_name"],
                                                  flags.split,
                                                  epoch, score))


def default_hparams_rand(flags, view):
    return {
        "view": view,
        "prot_dim": 8421,
        "comp_dim": 1024,
        "hdims": [3795, 2248, 2769, 2117],

        # weight initialization
        "kaiming_constant": 5,

        # dropout regs
        "dprob": 0.0739227,

        "tr_batch_size": 256,
        "val_batch_size": 512,
        "test_batch_size": 512,

        # optimizer params
        "optimizer": "rmsprop",
        "optimizer__sgd__weight_decay": 1e-4,
        "optimizer__sgd__nesterov": True,
        "optimizer__sgd__momentum": 0.9,
        "optimizer__sgd__lr": 1e-3,

        "optimizer__adam__weight_decay": 1e-4,
        "optimizer__adam__lr": 1e-3,

        "optimizer__rmsprop__lr": 0.000235395,
        "optimizer__rmsprop__weight_decay": 0.000146688,
        "optimizer__rmsprop__momentum": 0.00622082,
        "optimizer__rmsprop__centered": False
    }


def default_hparams_bopt(flags, comp_view, prot_view):
    return {
        "comp_view": comp_view,
        "hdims": [765, 2675],
        "output_dim": len(flags["tasks"]),

        # weight initialization
        "kaiming_constant": 5,

        # dropout regs
        "dprob": 0.02907481292488022,

        "tr_batch_size": 256,
        "val_batch_size": 128,
        "test_batch_size": 128,

        # optimizer params
        "optimizer": "adam",
        "optimizer__global__weight_decay": 0.0008599834489091449,
        "optimizer__global__lr": 0.00023239445781642333,
        "optimizer__adadelta__rho": 0.115873,

        "prot": {
            "model_type": prot_view,
            "vocab_size": flags["prot_vocab_size"],
            "window": 11,
            "dim": 8421 if prot_view == "psc" else 50,
            "pcnn_num_layers": 2
        },

        "weave": {
            "dim": 251,
            "update_pairs": False,
        },
        "gconv": {
            "dim": 512,
        },
        "ecfp8": {
            "dim": 1024,
        },
        "gnn": {
            "fingerprint_size": len(flags["gnn_fingerprint"]) if flags["gnn_fingerprint"] is not None else 0,
            "num_layers": 1,
            "dim": 160,
        }
    }


def get_hparam_config(flags, comp_view, prot_view):
    return {
        "comp_view": ConstantParam(comp_view),
        "hdims": DiscreteParam(min=256, max=5000, size=DiscreteParam(min=1, max=4)),
        "output_dim": ConstantParam(len(flags["tasks"])),

        # weight initialization
        "kaiming_constant": ConstantParam(5),  # DiscreteParam(min=2, max=9),

        # dropout regs
        "dprob": LogRealParam(min=-2),

        "tr_batch_size": CategoricalParam(choices=[128, 256]),
        "val_batch_size": ConstantParam(128),
        "test_batch_size": ConstantParam(128),

        # optimizer params
        "optimizer": CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
        "optimizer__global__weight_decay": LogRealParam(),
        "optimizer__global__lr": LogRealParam(),

        "prot": DictParam({
            "model_type": ConstantParam(prot_view),
            "vocab_size": ConstantParam(flags["prot_vocab_size"]),
            "window": ConstantParam(11),
            "dim": ConstantParam(8421) if prot_view == "psc" else DiscreteParam(min=5, max=50),
            "pcnn_num_layers": DiscreteParam(min=1, max=4)
        }),
        "weave": DictParam({
            "dim": DiscreteParam(min=64, max=512),
            "update_pairs": ConstantParam(False),
        }),
        "gconv": DictParam({
            "dim": DiscreteParam(min=64, max=512),
        }),
        "ecfp8": DictParam({
            "dim": ConstantParam(1024),
        }),
        "gnn": DictParam({
            "fingerprint_size": ConstantParam(len(flags["gnn_fingerprint"])) if flags[
                "gnn_fingerprint"] else ConstantParam(0),
            "num_layers": DiscreteParam(1, 4),
            "dim": DiscreteParam(min=64, max=512),
        })

        # # SGD
        # "optimizer__sgd__nesterov": CategoricalParam(choices=[True, False]),
        # "optimizer__sgd__momentum": LogRealParam(),
        # # "optimizer__sgd__lr": LogRealParam(),
        #
        # # ADAM
        # # "optimizer__adam__lr": LogRealParam(),
        # "optimizer__adam__amsgrad": CategoricalParam(choices=[True, False]),
        #
        # # Adadelta
        # # "optimizer__adadelta__lr": LogRealParam(),
        # # "optimizer__adadelta__weight_decay": LogRealParam(),
        # "optimizer__adadelta__rho": LogRealParam(),
        #
        # # Adagrad
        # # "optimizer__adagrad__lr": LogRealParam(),
        # "optimizer__adagrad__lr_decay": LogRealParam(),
        # # "optimizer__adagrad__weight_decay": LogRealParam(),
        #
        # # Adamax
        # # "optimizer__adamax__lr": LogRealParam(),
        # # "optimizer__adamax__weight_decay": LogRealParam(),
        #
        # # RMSprop
        # # "optimizer__rmsprop__lr": LogRealParam(),
        # # "optimizer__rmsprop__weight_decay": LogRealParam(),
        # "optimizer__rmsprop__momentum": LogRealParam(),
        # # "optimizer__rmsprop__centered": CategoricalParam(choices=[True, False])

    }


def get_hparam_file(cview, pview):
    return {'ecfp8-psc': 'bayopt_search_single_view_ecfp8_psc_dti_2019_11_19__01_22_15_gp_3000.csv',
            'weave-psc': 'bayopt_search_single_view_weave_psc_dti_2019_11_19__15_49_07_gp_3000.csv',
            'gconv-psc': 'bayopt_search_single_view_gconv_psc_dti_2019_11_20__23_23_12_gp_3000.csv',
            'gnn-psc': 'bayopt_search_single_view_gnn_psc_dti_2019_11_21__21_46_59_gp_3000.csv',
            'ecfp8-pcnna': 'bayopt_search_single_view_ecfp8_pcnna_dti_2019_11_23__20_23_56_gp_3000.csv',
            'weave-pcnna': 'bayopt_search_single_view_weave_pcnna_dti_2019_11_23__20_23_56_gp_3000.csv',
            'gconv-pcnna': 'bayopt_search_single_view_gconv_pcnna_dti_2019_11_23__20_23_56_gp_3000.csv',
            'ecfp8-rnn': 'bayopt_search_single_view_ecfp8_rnn_dti_2019_11_23__20_23_56_gp_3000.csv',
            'weave-rnn': 'bayopt_search_single_view_weave_rnn_dti_2019_11_23__20_23_56_gp_3000.csv',
            'gconv-rnn': 'bayopt_search_single_view_gconv_rnn_dti_2019_11_23__20_23_56_gp_3000.csv',
            'gnn-rnn': 'bayopt_search_single_view_gnn_rnn_dti_2019_11_23__20_23_56_gp_3000.csv',
            }.get(f'{cview.lower()}-{pview.lower()}', None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DTI with jova model training.")

    parser.add_argument("--dataset_name",
                        type=str,
                        default="davis",
                        help="Dataset name.")
    parser.add_argument("--dataset_file",
                        type=str,
                        help="Dataset file.")

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
    parser.add_argument('--split',
                        help='Splitting scheme to use. Options are: [warm, cold_drug, cold_target, cold_drug_target]',
                        action='append',
                        type=str,
                        dest='split_schemes'
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
    parser.add_argument('--prot_profile',
                        type=str,
                        help='A resource for retrieving embedding indexing profile of proteins.'
                        )
    parser.add_argument('--prot_vocab',
                        type=str,
                        help='A resource containing all N-gram segments/words constructed from the protein sequences.'
                        )
    parser.add_argument('--no_reload',
                        action="store_false",
                        dest='reload',
                        help='Whether datasets will be reloaded from existing ones or newly constructed.'
                        )
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
    parser.add_argument('--mp', '-mp', action='store_true', help="Multiprocessing option")

    args = parser.parse_args()
    procs = []
    use_mp = args.mp
    for split in args.split_schemes:
        flags = Flags()
        args_dict = args.__dict__
        for arg in args_dict:
            setattr(flags, arg, args_dict[arg])
        setattr(flags, "cv", True if flags.fold_num > 2 else False)
        setattr(flags, "views", [(cv, pv) for cv, pv in zip(args.comp_view, args.prot_view)])
        flags['split'] = split
        flags['predict_cold'] = split == 'cold_drug_target'
        flags['cold_drug'] = split == 'cold_drug'
        flags['cold_target'] = split == 'cold_target'
        flags['cold_drug_cluster'] = split == 'cold_drug_cluster'
        flags['split_warm'] = split == 'warm'
        if use_mp:
            p = mp.spawn(fn=main, args=(flags,), join=False)
            procs.append(p)
            # p.start()
        else:
            main(0, flags)
    for proc in procs:
        proc.join()
