# Author: bbrighttaer
# Project: jova
# Date: 10/17/19
# Time: 10:23 AM
# File: train_joint_dina_gan.py


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
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sch
from deepchem.trans import undo_transforms
from soek import RealParam, DictParam
from soek.bopt import BayesianOptSearchCV
from soek.params import ConstantParam, LogRealParam, DiscreteParam, CategoricalParam
from soek.rand import RandomSearchCV
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import jova.metrics as mt
from jova import cuda
from jova.data import batch_collator, get_data, load_proteins, DtiDataset
from jova.metrics import compute_model_performance
from jova.nn.layers import GraphConvLayer, GraphPool, Unsqueeze, GraphGather2D, ElementwiseBatchNorm
from jova.nn.models import GraphConvSequential, WeaveModel, NwayForward, JointAttention, Prot2Vec, ProteinRNN, \
    GraphNeuralNet2D, ProteinCNN2D, ProteinCNN
from jova.utils import Trainer, io
from jova.utils.args import WeaveLayerArgs, WeaveGatherArgs
from jova.utils.attn_helpers import MultimodalAttentionData
from jova.utils.io import load_pickle
from jova.utils.math import ExpAverage, Count
from jova.utils.sim_data import DataNode
from jova.utils.tb import TBMeanTracker
from jova.utils.train_helpers import count_parameters, GradStats, FrozenModels

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1, 8, 64]

# seeds = [123, 124, 125]

check_data = False

torch.cuda.set_device(0)

use_ecfp8 = False
use_weave = False
use_gconv = False
use_prot = True
use_gnn = True

joint_attention_data = MultimodalAttentionData()


def create_ecfp_net(hparams):
    model = nn.Sequential(Unsqueeze(dim=0))
    return model


def create_prot_net(hparams, model_type, protein_profile, protein_embeddings, frozen_models_hook=None):
    # assert protein_profile is not None, "Protein profile has to be supplied"
    # assert protein_embeddings is not None, "Pre-trained protein embeddings are required"

    # model_type = hparams["prot"]["model_type"].lower()
    valid_opts = ["rnn", "psc", "p2v", "pcnn", "pcnn2d"]
    assert (model_type in valid_opts), "Valid protein types: {}".format(str(valid_opts))
    if model_type == "rnn":
        # pt_embeddings = create_torch_embeddings(frozen_models_hook, protein_embeddings)
        model = nn.Sequential(Prot2Vec(protein_profile=protein_profile,
                                       vocab_size=hparams["prot"]["vocab_size"],
                                       embedding_dim=hparams["prot"]["embedding_dim"],
                                       batch_first=False),
                              ProteinRNN(in_dim=hparams["prot"]["embedding_dim"] * hparams["prot"]["window"],
                                         hidden_dim=hparams["prot"]["rnn_hidden_state_dim"],
                                         dropout=hparams["dprob"],
                                         batch_first=False))
    elif model_type == "p2v":
        # pt_embeddings = create_torch_embeddings(frozen_models_hook, protein_embeddings)
        model = Prot2Vec(protein_profile=protein_profile,
                         vocab_size=hparams["prot"]["vocab_size"],
                         embedding_dim=hparams["prot"]["embedding_dim"],
                         batch_first=False)
    elif model_type == "pcnn":
        # pt_embeddings = create_torch_embeddings(frozen_models_hook, protein_embeddings)
        model = nn.Sequential(Prot2Vec(protein_profile=protein_profile,
                                       vocab_size=hparams["prot"]["vocab_size"],
                                       embedding_dim=hparams["prot"]["embedding_dim"],
                                       batch_first=False),
                              ProteinCNN(dim=hparams["prot"]["embedding_dim"],
                                         window=hparams["prot"]["window"],
                                         activation="relu",
                                         pooling_dim=0,
                                         num_layers=hparams["prot"]["pcnn_num_layers"]),
                              Unsqueeze(dim=0))
    elif model_type == "pcnn2d":
        # pt_embeddings = create_torch_embeddings(frozen_models_hook, protein_embeddings)
        model = nn.Sequential(Prot2Vec(protein_profile=protein_profile,
                                       vocab_size=hparams["prot"]["vocab_size"],
                                       embedding_dim=hparams["prot"]["embedding_dim"],
                                       batch_first=False),
                              ProteinCNN2D(dim=hparams["prot"]["embedding_dim"],
                                           window=hparams["prot"]["window"],
                                           num_layers=hparams["prot"]["pcnn_num_layers"]))
    elif model_type == "psc":
        model = nn.Sequential(Unsqueeze(dim=0))
    return model


def create_weave_net(hparams):
    weave_args = (
        WeaveLayerArgs(n_atom_input_feat=75,
                       n_pair_input_feat=14,
                       n_atom_output_feat=50,
                       # n_atom_output_feat=hparams["weave"]["dim"],
                       n_pair_output_feat=50,
                       n_hidden_AA=50,
                       n_hidden_PA=50,
                       n_hidden_AP=50,
                       n_hidden_PP=50,
                       update_pair=hparams["weave"]["update_pairs"],
                       activation='relu',
                       batch_norm=True,
                       dropout=hparams["dprob"]
                       ),
        WeaveLayerArgs(n_atom_input_feat=50,
                       n_pair_input_feat=14,
                       n_atom_output_feat=hparams["weave"]["dim"],
                       n_pair_output_feat=50,
                       n_hidden_AA=50,
                       n_hidden_PA=50,
                       n_hidden_AP=50,
                       n_hidden_PP=50,
                       update_pair=hparams["weave"]["update_pairs"],
                       batch_norm=True,
                       dropout=hparams["dprob"],
                       activation='relu'),
    )
    wg_args = WeaveGatherArgs(conv_out_depth=hparams["weave"]["dim"], gaussian_expand=True,
                              n_depth=hparams["weave"]["dim"])
    weave_model = WeaveModel(weave_args, weave_gath_arg=wg_args, weave_type='2D')
    model = nn.Sequential(weave_model)
    return model


def create_gconv_net(hparams):
    dim = hparams["gconv"]["dim"]
    gconv_model = GraphConvSequential(GraphConvLayer(in_dim=75, out_dim=64),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      GraphPool(),

                                      GraphConvLayer(in_dim=64, out_dim=64),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      GraphPool(),

                                      nn.Linear(in_features=64, out_features=dim),
                                      nn.BatchNorm1d(dim),
                                      nn.ReLU(),
                                      nn.Dropout(hparams["dprob"]),
                                      GraphGather2D(activation='nonsat'))

    model = nn.Sequential(gconv_model)
    return model


def create_gnn_net(hparams):
    dim = hparams["gnn"]["dim"]
    gnn_model = GraphNeuralNet2D(num_fingerprints=hparams["gnn"]["fingerprint_size"], embedding_dim=dim,
                                 num_layers=hparams["gnn"]["num_layers"])
    return nn.Sequential(gnn_model,
                         nn.Linear(dim, dim),
                         ElementwiseBatchNorm(dim),
                         nn.ReLU(),
                         nn.Dropout(hparams["dprob"]))


def create_integrated_net(hparams, protein_profile, protein_embeddings):
    # Convenient way of keeping track of models to be frozen during (or at the initial stages) training.
    frozen_models = FrozenModels()

    # N-way forward propagation
    views = {}
    seg_dims = []

    # compound models
    if use_ecfp8:
        views["ecfp8"] = create_ecfp_net(hparams)
        seg_dims.append(hparams["ecfp8"]["dim"])
        joint_attention_data._labels.append("ecfp8")
    if use_weave:
        views["weave"] = create_weave_net(hparams)
        seg_dims.append(hparams["weave"]["dim"])
        joint_attention_data._labels.append("weave")
    if use_gconv:
        views["gconv"] = create_gconv_net(hparams)
        seg_dims.append(hparams["gconv"]["dim"])
        joint_attention_data._labels.append("gconv")
    if use_gnn:
        views["gnn"] = create_gnn_net(hparams)
        seg_dims.append(hparams["gnn"]["dim"])
        joint_attention_data._labels.append("gnn")

    # protein models
    for m_type in hparams["prot"]["model_types"]:
        views[m_type] = create_prot_net(hparams, m_type, protein_profile, protein_embeddings, frozen_models)
        if m_type == "psc":
            seg_dims.append(hparams["prot"]["psc_dim"])
        elif m_type == "rnn":
            seg_dims.append(hparams["prot"]["rnn_hidden_state_dim"])
        elif m_type == "p2v":
            seg_dims.append(hparams["prot"]["embedding_dim"] * hparams["prot"]["window"])
        elif m_type in ["pcnn", "pcnn2d"]:
            seg_dims.append(hparams["prot"]["embedding_dim"])

        # register the view of the protein for recording attention info
        joint_attention_data._labels.append(m_type)

    layers = [NwayForward(models=views.values())]

    func_callback = joint_attention_data.joint_attn_forward_hook if hparams["explain_mode"] else None
    layers.append(JointAttention(d_dims=seg_dims, latent_dim=hparams["latent_dim"], num_heads=hparams["attn_heads"],
                                 num_layers=hparams["attn_layers"], dprob=hparams["dprob"],
                                 attn_hook=func_callback))

    p = len(views) * hparams["latent_dim"]
    for dim in hparams["lin_dims"]:
        layers.append(nn.Linear(p, dim))
        layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(hparams["dprob"]))
        p = dim

    # Output layer
    layers.append(nn.Linear(in_features=p, out_features=hparams["output_dim"]))

    # Build model
    model = nn.Sequential(*layers)

    return model, frozen_models


class Jova(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset, protein_profile, protein_embeddings,
                   cuda_devices=None, mode="regression"):

        # create networks
        model, frozen_models = create_integrated_net(hparams, protein_profile, protein_embeddings)
        print("Number of trainable parameters: model={}".format(count_parameters(model)))
        if cuda:
            model = model.cuda()

        # data loaders
        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=hparams["tr_batch_size"],
                                       shuffle=True,
                                       collate_fn=lambda x: x)
        val_data_loader = DataLoader(dataset=val_dataset,
                                     batch_size=1 if hparams["explain_mode"] else hparams["val_batch_size"],
                                     shuffle=False,
                                     collate_fn=lambda x: x)
        test_data_loader = None
        if test_dataset is not None:
            test_data_loader = DataLoader(dataset=test_dataset,
                                          batch_size=1 if hparams["explain_mode"] else hparams["test_batch_size"],
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
        return model, optimizer, {"train": train_data_loader, "val": val_data_loader,
                                  "test": test_data_loader}, metrics, hparams["prot"]["model_types"], frozen_models

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
            data = {"train": train_dataset, "val": valid_dataset, "test": None}
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
    def train(eval_fn, model, optimizer, data_loaders, metrics, prot_model_types, frozen_models, transformers_dict,
              prot_desc_dict, tasks, n_iters=5000, sim_data_node=None, epoch_ckpt=(2, 1.0), tb_writer=None):
        tb_writer = tb_writer()
        start = time.time()
        best_model_wts = model.state_dict()
        best_score = -10000
        best_epoch = -1
        terminate_training = False
        e_avg = ExpAverage(.01)
        n_epochs = n_iters // len(data_loaders["train"])
        grad_stats = GradStats(model, beta=0.)

        # learning rate decay schedulers
        scheduler = sch.StepLR(optimizer, step_size=500, gamma=0.01)

        # pred_loss functions
        prediction_criterion = nn.MSELoss()

        # sub-nodes of sim data resource
        loss_lst = []
        train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        train_scores_lst = []
        train_scores_node = DataNode(label="train_score", data=train_scores_lst)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)
        loss_lst = []
        loss_node = DataNode(label="generator_loss", data=loss_lst)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [train_loss_node, train_scores_node, metrics_node, scores_node, loss_node]

        try:
            # Main training loop
            tb_idx = Count()
            for epoch in range(n_epochs):
                if terminate_training:
                    print("Terminating training...")
                    break
                for phase in ["train", "val"]:
                    # ensure these models are frozen at all times
                    for m in frozen_models:
                        m.eval()

                    if phase == "train":
                        print("Training....")
                        # Adjust the learning rate.
                        # scheduler_gen.step()
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
                        with grad_stats:
                            for batch in tqdm(data_loaders[phase]):
                                batch_size, data = batch_collator(batch, prot_desc_dict, spec={"ecfp8": use_ecfp8,
                                                                                               "weave": use_weave,
                                                                                               "gconv": use_gconv,
                                                                                               "gnn": use_gnn},
                                                                  cuda_prot=True)

                                # organize the data for each view.
                                Xs = {}
                                Ys = {}
                                Ws = {}
                                for view_name in data:
                                    view_data = data[view_name]
                                    if view_name == "gconv":
                                        x = ((view_data[0][0], batch_size), view_data[0][1], view_data[0][2])
                                        Xs["gconv"] = x
                                    else:
                                        Xs[view_name] = view_data[0]
                                    Ys[view_name] = np.array([k for k in view_data[1]], dtype=np.float)
                                    Ws[view_name] = np.array([k for k in view_data[2]], dtype=np.float)

                                optimizer.zero_grad()

                                # forward propagation
                                # track history if only in train
                                with torch.set_grad_enabled(phase == "train"):
                                    Ys = {k: Ys[k] for k in Ys}
                                    # Ensure matching labels across views.
                                    for j in range(1, len(Ys.values())):
                                        assert (list(Ys.values())[j - 1] == list(Ys.values())[j]).all()

                                    y = Ys[list(Xs.keys())[0]]
                                    w = Ws[list(Xs.keys())[0]]

                                    # protein data in batch
                                    protein_xs = []
                                    for m_type in prot_model_types:
                                        if m_type in ["p2v", "rnn", "pcnn", "pcnn2d"]:
                                            protein_xs.append(Xs[list(Xs.keys())[0]][2])
                                        elif m_type == "psc":
                                            protein_xs.append(Xs[list(Xs.keys())[0]][1])

                                    # compound data in batch
                                    X = []
                                    if use_ecfp8:
                                        X.append(Xs["ecfp8"][0])
                                    if use_weave:
                                        X.append(Xs["weave"][0])
                                    if use_gconv:
                                        X.append(Xs["gconv"][0])
                                    if use_gnn:
                                        X.append(Xs["gnn"][0])

                                    # merge compound and protein list
                                    X = X + protein_xs

                                    # forward pass
                                    outputs = model(X)

                                    # compute loss
                                    target = torch.from_numpy(y).float()
                                    weights = torch.from_numpy(w).float()
                                    if cuda:
                                        target = target.cuda()
                                        weights = weights.cuda()
                                    outputs = outputs * weights
                                    loss = prediction_criterion(outputs, target)

                                if phase == "train":
                                    tracker.track("train/train_pred_loss", loss.item(), tb_idx.IncAndGet())

                                    # backward pass
                                    loss.backward()
                                    optimizer.step()

                                    # for epoch stats
                                    epoch_losses.append(loss.item())

                                    # for sim data resource
                                    loss_lst.append(loss.item())

                                    print("\tEpoch={}/{}, batch={}/{}, pred_loss={:.4f}".format(
                                        epoch + 1, n_epochs,
                                        i + 1,
                                        len(data_loaders[phase]), loss.item()))

                                    # monitor train data score
                                    eval_dict = {}
                                    score = eval_fn(eval_dict, y, outputs, w, metrics, tasks,
                                                    transformers_dict[list(Xs.keys())[0]])
                                    train_scores_lst.append(score)
                                    tracker.track("train/score", score, tb_idx.i)
                                else:
                                    if str(loss.item()) != "nan":  # useful in hyperparameter search
                                        tracker.track("val/val_pred_loss", loss.item(), tb_idx.i)
                                        eval_dict = {}
                                        score = eval_fn(eval_dict, y, outputs, w, metrics, tasks,
                                                        transformers_dict[list(Xs.keys())[0]])

                                        # TBoard info
                                        tracker.track('val/score', score, tb_idx.i)
                                        for k in eval_dict:
                                            tracker.track('val/{}'.format(k), eval_dict[k], tb_idx.i)

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
                                data_size += batch_size
                        # End of mini=batch iterations.

                    if phase == "train":
                        ep_loss = np.nanmean(epoch_losses)
                        e_avg.update(ep_loss)
                        if epoch % (epoch_ckpt[0] - 1) == 0 and epoch > 0:
                            if e_avg.value > epoch_ckpt[1]:
                                terminate_training = True
                        print("\nPhase: {}, avg task pred_loss={:.4f}, ".format(phase, np.nanmean(epoch_losses)))
                        scheduler.step()
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
        return model, best_score, best_epoch

    @staticmethod
    def evaluate_model(eval_fn, model, model_dir, model_name, data_loaders, metrics, transformers_dict,
                       prot_desc_dict, prot_model_types, tasks, sim_data_node):
        # load saved model and put in evaluation mode
        model.load_state_dict(io.load_model(model_dir, model_name, dvc=torch.device("cuda:2")))
        model.eval()

        print("Model evaluation...")
        start = time.time()

        # sub-nodes of sim data resource
        # loss_lst = []
        # train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)
        predicted_vals = []
        true_vals = []
        model_preds_node = DataNode(label="model_predictions", data={"y": true_vals,
                                                                     "y_pred": predicted_vals})
        attn_ranking = []
        attn_ranking_node = DataNode(label="attn_ranking", data=attn_ranking)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [metrics_node, scores_node, model_preds_node, attn_ranking_node]

        # Main evaluation loop
        i = 0
        for batch in tqdm(data_loaders['val']):
            batch_size, data = batch_collator(batch, prot_desc_dict, spec={"ecfp8": use_ecfp8,
                                                                           "weave": use_weave,
                                                                           "gconv": use_gconv,
                                                                           "gnn": use_gnn},
                                              cuda_prot=True)

            # attention x data for analysis
            attn_data_x = {}

            # organize the data for each view.
            Xs = {}
            Ys = {}
            Ws = {}
            for view_name in data:
                view_data = data[view_name]
                if view_name == "gconv":
                    x = ((view_data[0][0], batch_size), view_data[0][1], view_data[0][2])
                    Xs["gconv"] = x
                else:
                    Xs[view_name] = view_data[0]
                Ys[view_name] = np.array([k for k in view_data[1]], dtype=np.float)
                Ws[view_name] = np.array([k for k in view_data[2]], dtype=np.float)
                attn_data_x[view_name] = view_data[0][3]

            with torch.set_grad_enabled(False):
                Ys = {k: Ys[k].astype(np.float) for k in Ys}
                # Ensure corresponding pairs
                for j in range(1, len(Ys.values())):
                    assert (list(Ys.values())[j - 1] == list(Ys.values())[j]).all()

                y_true = Ys[list(Xs.keys())[0]]
                w = Ws[list(Xs.keys())[0]]

                # protein data in batch
                protein_xs = []
                for m_type in prot_model_types:
                    if m_type in ["p2v", "rnn", "pcnn", "pcnn2d"]:
                        protein_xs.append(Xs[list(Xs.keys())[0]][2])
                    elif m_type == "psc":
                        protein_xs.append(Xs[list(Xs.keys())[0]][1])
                    attn_data_x[m_type] = Xs[list(Xs.keys())[0]][2]

                # compound data in batch
                X = []
                if use_ecfp8:
                    X.append(Xs["ecfp8"][0])
                if use_weave:
                    X.append(Xs["weave"][0])
                if use_gconv:
                    X.append(Xs["gconv"][0])
                if use_gnn:
                    X.append(Xs["gnn"][0])

                # merge compound and protein list
                X = X + protein_xs

                # forward propagation
                y_predicted = model(X)

                # apply transformers
                transformer = transformers_dict[list(Xs.keys())[0]]
                predicted_vals.extend(np_to_plot_data(undo_transforms(y_predicted.cpu().detach().numpy(), transformer)))
                true_vals.extend(np_to_plot_data(undo_transforms(y_true, transformer)))

            eval_dict = {}
            score = eval_fn(eval_dict, y_true, y_predicted, w, metrics, tasks, transformer)

            # for sim data resource
            scores_lst.append(score)
            for m in eval_dict:
                if m in metrics_dict:
                    metrics_dict[m].append(eval_dict[m])
                else:
                    metrics_dict[m] = [eval_dict[m]]

            print("\nbatch={}/{}, evaluation results= {}, score={}".format(i + 1, len(data_loaders['val']), eval_dict,
                                                                           score))

            i += 1
        # End of mini=batch iterations.

        duration = time.time() - start
        print('\nModel evaluation duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

    @staticmethod
    def explain_model(model, model_dir, model_name, data_loaders, transformers_dict, prot_desc_dict, prot_model_types,
                      sim_data_node, max_print=10, k=10):
        # load saved model and put in evaluation mode
        model.load_state_dict(io.load_model(model_dir, model_name, dvc=torch.device("cuda:2")))
        model.eval()

        print("Model explanation...")
        start = time.time()

        # sub-nodes of sim data resource
        attn_ranking = []
        attn_ranking_node = DataNode(label="attn_ranking", data=attn_ranking)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [attn_ranking_node]

        # Main evaluation loop
        i = 0
        for batch in tqdm(data_loaders['val']):
            if i == max_print:
                print('\nMaximum number [%d] of samples limit reached. Terminating...' % i)
                break
            i += 1

            batch_size, data = batch_collator(batch, prot_desc_dict, spec={"ecfp8": use_ecfp8,
                                                                           "weave": use_weave,
                                                                           "gconv": use_gconv,
                                                                           "gnn": use_gnn},
                                              cuda_prot=True)

            # attention x data for analysis
            attn_data_x = {}

            # organize the data for each view.
            Xs = {}
            Ys = {}
            Ws = {}
            for view_name in data:
                view_data = data[view_name]
                if view_name == "gconv":
                    x = ((view_data[0][0], batch_size), view_data[0][1], view_data[0][2])
                    Xs["gconv"] = x
                else:
                    Xs[view_name] = view_data[0]
                Ys[view_name] = np.array([k for k in view_data[1]], dtype=np.float)
                Ws[view_name] = np.array([k for k in view_data[2]], dtype=np.float)
                attn_data_x[view_name] = view_data[0][3]

            with torch.set_grad_enabled(False):
                Ys = {k: Ys[k].astype(np.float) for k in Ys}
                # Ensure corresponding pairs
                for j in range(1, len(Ys.values())):
                    assert (list(Ys.values())[j - 1] == list(Ys.values())[j]).all()

                y_true = Ys[list(Xs.keys())[0]]
                w = Ws[list(Xs.keys())[0]]

                # protein data in batch
                protein_xs = []
                for m_type in prot_model_types:
                    if m_type in ["p2v", "rnn", "pcnn", "pcnn2d"]:
                        protein_xs.append(Xs[list(Xs.keys())[0]][2])
                    elif m_type == "psc":
                        protein_xs.append(Xs[list(Xs.keys())[0]][1])
                    attn_data_x[m_type] = Xs[list(Xs.keys())[0]][2]

                # compound data in batch
                X = []
                if use_ecfp8:
                    X.append(Xs["ecfp8"][0])
                if use_weave:
                    X.append(Xs["weave"][0])
                if use_gconv:
                    X.append(Xs["gconv"][0])
                if use_gnn:
                    X.append(Xs["gnn"][0])

                # merge compound and protein list
                X = X + protein_xs

                # register corresponding joint attention data before forward pass
                joint_attention_data.register_data(attn_data_x)

                # forward propagation
                y_predicted = model(X)

                # get segments ranking
                transformer = transformers_dict[list(Xs.keys())[0]]
                rank_results = {'y_pred': np_to_plot_data(undo_transforms(y_predicted.cpu().detach().numpy(),
                                                                          transformer)),
                                'y_true': np_to_plot_data(undo_transforms(y_true, transformer)),
                                'attn_ranking': joint_attention_data.get_rankings(k)}
                attn_ranking.append(rank_results)
        # End of mini=batch iterations.

        duration = time.time() - start
        print('\nModel explanation duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))


def np_to_plot_data(y):
    y = y.squeeze()
    if y.shape == ():
        return [float(y)]
    else:
        return y.squeeze().tolist()


def main(flags):
    comp_lbl = []
    if use_ecfp8:
        comp_lbl.append("ecfp8")
    if use_weave:
        comp_lbl.append("weave")
    if use_gconv:
        comp_lbl.append("gconv")
    if use_gnn:
        comp_lbl.append("gnn")
    comp_lbl = '_'.join(comp_lbl)
    flags["prot_model_types"] = ["rnn"]
    sim_label = "integrated_view_attn_no_gan_" + ('_'.join(flags["prot_model_types"])) + '_' + comp_lbl
    print("CUDA={}, view={}".format(cuda, sim_label))

    # Simulation data resource tree
    split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
        flags["cold_drug"] else "None"
    dataset_lbl = flags["dataset"]

    if flags["eval"]:
        mode = "eval"
    elif flags["explain"]:
        mode = "explain"
    else:
        mode = "train"

    node_label = "{}_{}_{}_{}_{}".format(dataset_lbl, sim_label, split_label, mode, date_label)
    sim_data = DataNode(label=node_label)
    nodes_list = []
    sim_data.data = nodes_list

    num_cuda_dvcs = torch.cuda.device_count()
    cuda_devices = None if num_cuda_dvcs == 1 else [i for i in range(1, num_cuda_dvcs)]

    # Runtime Protein stuff
    prot_desc_dict, prot_seq_dict = load_proteins(flags['prot_desc_path'])
    prot_profile = load_pickle(file_name=flags.prot_profile)
    prot_vocab = load_pickle(file_name=flags.prot_vocab)
    pretrained_embeddings = None  # load_numpy_array(flags.protein_embeddings)
    flags["prot_vocab_size"] = len(prot_vocab)

    # set joint attention hook's protein information
    joint_attention_data.protein_profile = prot_profile
    joint_attention_data.protein_vocabulary = prot_vocab
    joint_attention_data.protein_sequences = prot_seq_dict

    # For searching over multiple seeds
    hparam_search = None

    for seed in seeds:
        summary_writer_creator = lambda: SummaryWriter(log_dir="tb_runs/{}_{}_{}/".format(sim_label, seed,
                                                                                          dt.now().strftime(
                                                                                              "%Y_%m_%d__%H_%M_%S")))

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

        # Data
        flags["gnn_fingerprint"] = None
        if use_ecfp8:
            data_dict["ecfp8"] = get_data("ECFP8", flags, prot_sequences=prot_seq_dict, seed=seed)
            transformers_dict["ecfp8"] = data_dict["ecfp8"][2]
        if use_weave:
            data_dict["weave"] = get_data("Weave", flags, prot_sequences=prot_seq_dict, seed=seed)
            transformers_dict["weave"] = data_dict["weave"][2]
        if use_gconv:
            data_dict["gconv"] = get_data("GraphConv", flags, prot_sequences=prot_seq_dict, seed=seed)
            transformers_dict["gconv"] = data_dict["gconv"][2]
        if use_gnn:
            data_dict["gnn"] = get_data("GNN", flags, prot_sequences=prot_seq_dict, seed=seed)
            transformers_dict["gnn"] = data_dict["gnn"][2]
            flags["gnn_fingerprint"] = data_dict["gnn"][3]

        tasks = data_dict[list(data_dict.keys())[0]][0]
        # multi-task or single task is determined by the number of tasks w.r.t. the dataset loaded
        flags["tasks"] = tasks

        trainer = Jova()

        if flags["cv"]:
            k = flags["fold_num"]
            print("{}, {}-Prot: Training scheme: {}-fold cross-validation".format(tasks, sim_label, k))
        else:
            k = 1
            print("{}, {}-Prot: Training scheme: train, validation".format(tasks, sim_label)
                  + (", test split" if flags['test'] else " split"))

        if check_data:
            verify_multiview_data(data_dict)
        else:
            if flags["hparam_search"]:
                print("Hyperparameter search enabled: {}".format(flags["hparam_search_alg"]))

                # arguments to callables
                extra_init_args = {"mode": "regression",
                                   "cuda_devices": cuda_devices,
                                   "protein_profile": prot_profile,
                                   "protein_embeddings": pretrained_embeddings}
                extra_data_args = {"flags": flags,
                                   "data_dict": data_dict}
                extra_train_args = {"transformers_dict": transformers_dict,
                                    "prot_desc_dict": prot_desc_dict,
                                    "tasks": tasks,
                                    "n_iters": 3000,
                                    "tb_writer": summary_writer_creator}

                hparams_conf = get_hparam_config(flags)

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
                                               eval_fn=trainer.evaluate,
                                               save_model_fn=io.save_model,
                                               init_args=extra_init_args,
                                               data_args=extra_data_args,
                                               train_args=extra_train_args,
                                               data_node=data_node,
                                               split_label=split_label,
                                               sim_label=sim_label,
                                               minimizer=min_opt,
                                               dataset_label=dataset_lbl,
                                               results_file="{}_{}_dti_{}_{}.csv".format(
                                                   flags["hparam_search_alg"], sim_label, date_label, min_opt))

                stats = hparam_search.fit(model_dir="models", model_name="".join(tasks), max_iter=40, seed=seed)
                print(stats)
                print("Best params = {}".format(stats.best(m="max")))
            else:
                invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node, sim_label,
                             prot_profile, pretrained_embeddings, prot_vocab, summary_writer_creator)

    # save simulation data resource tree to file.
    sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node, view, protein_profile,
                 protein_embeddings, prot_vocab, tb_writer):
    hyper_params = default_hparams_bopt(flags)
    # Initialize the model and other related entities for training.
    if flags["cv"]:
        folds_data = []
        data_node.data = folds_data
        data_node.label = data_node.label + "cv"
        for k in range(flags["fold_num"]):
            k_node = DataNode(label="fold-%d" % k)
            folds_data.append(k_node)
            start_fold(k_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
                       transformers_dict, view, protein_profile, protein_embeddings, prot_vocab, k, tb_writer)
    else:
        start_fold(data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
                   transformers_dict, view, protein_profile, protein_embeddings, prot_vocab, tb_writer)


def start_fold(sim_data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
               transformers_dict, view, protein_profile, protein_embeddings, protein_vocab, k=None,
               tb_writer=None):
    data = trainer.data_provider(k, flags, data_dict)
    model, optimizer, data_loaders, metrics, \
    prot_model_types, frozen_models = trainer.initialize(hparams=hyper_params, train_dataset=data["train"],
                                                         protein_profile=protein_profile,
                                                         protein_embeddings=protein_embeddings,
                                                         val_dataset=data["val"], test_dataset=data["test"])
    if flags["eval"]:
        trainer.evaluate_model(trainer.evaluate, model, flags["model_dir"], flags["eval_model_name"],
                               data_loaders, metrics, transformers_dict, prot_desc_dict, prot_model_types, tasks,
                               sim_data_node)
    elif flags["explain"]:
        trainer.explain_model(model, flags["model_dir"], flags["eval_model_name"], data_loaders, transformers_dict,
                              prot_desc_dict, prot_model_types, sim_data_node)
    else:
        # Train the model
        model, score, epoch = trainer.train(trainer.evaluate, model, optimizer, data_loaders, metrics, prot_model_types,
                                            frozen_models, transformers_dict, prot_desc_dict, tasks, n_iters=10000,
                                            sim_data_node=sim_data_node, tb_writer=tb_writer)
        # Save the model.
        split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
            flags["cold_drug"] else "None"
        io.save_model(model, flags["model_dir"],
                      "{}_{}_{}_{}_{}_{:.4f}".format(flags["dataset"], view, flags["model_name"], split_label, epoch,
                                                     score))


def default_hparams_rand(flags):
    return {
        "prot_vocab_size": flags["prot_vocab_size"],
        "attn_heads": 1,
        "n_ways": 3,
        "proj_out_dim": 512,

        # weight initialization
        "kaiming_constant": 5,

        # dropout regs
        "dprob": 0.0739227,

        "tr_batch_size": 256,
        "val_batch_size": 128,
        "test_batch_size": 128,

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
        "optimizer__rmsprop__centered": False,

        "prot": {
            "model_type": "Identity",
            "dim": 8421,
        },
        "weave": {
            "dim": 50,
            "update_pairs": False,
        },
        "gconv": {
            "dim": 128,
        },
        "ecfp": {
            "dim": 1024,
        }
    }


def default_hparams_bopt(flags):
    """
    protein model types:
    --------------------------------------------------------------------------------------------
    short name  | full name
    ------------|-------------------------------------------------------------------------------
    psc         | Protein Sequence Composition
    ------------|-------------------------------------------------------------------------------
    p2v         | Protein to Vector / Embeddings using n-gram amino acid 'words'.
    ------------|-------------------------------------------------------------------------------
    rnn         | Uses embeddings and an RNN variant (e.g. LSTM) to learn protein features.
    ------------|-------------------------------------------------------------------------------
    pcnn        | Protein CNN: https://academic.oup.com/bioinformatics/article/35/2/309/5050020
                | The final output is a 1D vector for each protein in a batch.
    ------------|-------------------------------------------------------------------------------
    pcnn2d      | A variant of PCNN that returns a 2D tensor for each protein in a batch.
    ------------|-------------------------------------------------------------------------------
    NOTE: All protein models, except 'psc' and 'p2v', use embeddings from the :class:Prot2Vec module.
    """
    return {
        "explain_mode": flags.explain,
        "attn_heads": 2,
        "attn_layers": 1,
        "lin_dims": [1033, 1481, 1800],
        "output_dim": len(flags.tasks),
        "latent_dim": 512,

        # weight initialization
        "kaiming_constant": 5,

        # dropout
        "dprob": 0.2,

        "tr_batch_size": 256,
        "val_batch_size": 128,
        "test_batch_size": 128,

        # optimizer params
        "optimizer": "adamax",
        "optimizer__global__weight_decay": 0.0007,
        "optimizer__global__lr": 0.0004,
        "prot": {
            "model_types": flags["prot_model_types"],
            "vocab_size": flags["prot_vocab_size"],
            "window": 11,
            "pcnn_num_layers": 2,
            "embedding_dim": 33,
            "psc_dim": 8421,
            "rnn_hidden_state_dim": 27
        },
        "weave": {
            "dim": 50,
            "update_pairs": False,
        },
        "gconv": {
            "dim": 512,
        },
        "ecfp8": {
            "dim": 1024,
        },
        "gnn": {
            "fingerprint_size": len(flags["gnn_fingerprint"]) if use_gnn else 0,
            "num_layers": 3,
            "dim": 100,
        }
    }


def get_hparam_config(flags):
    return {
        "explain_mode": ConstantParam(flags.explain),
        "attn_heads": CategoricalParam([1, 2, 4, 8, 16]),
        "attn_layers": DiscreteParam(min=1, max=3),
        "lin_dims": DiscreteParam(min=64, max=2048, size=DiscreteParam(min=1, max=3)),
        "output_dim": ConstantParam(len(flags.tasks)),
        "latent_dim": CategoricalParam(choices=[256, 512]),

        # weight initialization
        "kaiming_constant": ConstantParam(5),

        # dropout
        "dprob": RealParam(0.1, max=0.5),

        "tr_batch_size": CategoricalParam(choices=[128, 256]),
        "val_batch_size": ConstantParam(128),
        "test_batch_size": ConstantParam(128),

        # optimizer params
        "optimizer": CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
        "optimizer__global__weight_decay": LogRealParam(),
        "optimizer__global__lr": LogRealParam(),

        "prot": DictParam({
            "model_types": ConstantParam(flags["prot_model_types"]),
            "vocab_size": ConstantParam(flags["prot_vocab_size"]),
            "window": ConstantParam(11),
            "pcnn_num_layers": DiscreteParam(min=1, max=4),
            "embedding_dim": DiscreteParam(min=5, max=50),
            "psc_dim": ConstantParam(8421),
            "rnn_hidden_state_dim": DiscreteParam(min=5, max=50)
        }),
        "weave": ConstantParam({
            "dim": DiscreteParam(min=64, max=512),
            "update_pairs": ConstantParam(False),
        }),
        "gconv": DictParam({
            "dim": DiscreteParam(min=64, max=512),
        }),
        "ecfp8": DictParam({
            "dim": ConstantParam(1024),
        }),
        "gnn": ConstantParam({
            "fingerprint_size": ConstantParam(len(flags["gnn_fingerprint"]) if use_gnn else 0),
            "num_layers": DiscreteParam(1, 4),
            "dim": DiscreteParam(min=64, max=512),
        })
    }


def verify_multiview_data(data_dict, cv_data=True):
    if cv_data:
        ecfp8_data = data_dict["ecfp8"][1][0][0]
        weave_data = data_dict["weave"][1][0][0]
        gconv_data = data_dict["gconv"][1][0][0]
        gnn_data = data_dict["gnn"][1][0][0]
    else:
        ecfp8_data = data_dict["ecfp8"][1][0]
        weave_data = data_dict["weave"][1][0]
        gconv_data = data_dict["gconv"][1][0]
        gnn_data = data_dict["gnn"][1][0]
    corr = []
    for i in range(100):
        print("-" * 100)
        ecfp8 = "mol={}, prot={}, y={}".format(ecfp8_data.X[i][0].smiles, ecfp8_data.X[i][1].get_name(),
                                               ecfp8_data.y[i])
        print("ecfp8:", ecfp8)
        weave = "mol={}, prot={}, y={}".format(weave_data.X[i][0].smiles, weave_data.X[i][1].get_name(),
                                               weave_data.y[i])
        print("weave:", weave)
        gconv = "mol={}, prot={}, y={}".format(gconv_data.X[i][0].smiles, gconv_data.X[i][1].get_name(),
                                               gconv_data.y[i])
        print("gconv:", gconv)
        gnn = "mol={}, prot={}, y={}".format(gnn_data.X[i][0].smiles, gnn_data.X[i][1].get_name(),
                                             gnn_data.y[i])
        print("gnn:", gnn)
        print('#' * 100)
        corr.append(ecfp8 == weave == gconv == gnn)
    print(corr)


class Flags(object):
    # enables using either object referencing or dict indexing to retrieve user passed arguments of flag objects.
    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        setattr(self, key, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DTI with jova model training.")

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
    parser.add_argument('--prot_profile',
                        type=str,
                        help='A resource for retrieving embedding indexing profile of proteins.'
                        )
    parser.add_argument('--prot_vocab',
                        type=str,
                        help='A resource containing all N-gram segments/words constructed from the protein sequences.'
                        )
    parser.add_argument('--prot_embeddings',
                        type=str,
                        dest="protein_embeddings",
                        help='Numpy array file containing the pretrained protein "words" embeddings'
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
    parser.add_argument("--eval",
                        action="store_true",
                        help="If true, a saved model is loaded and evaluated")
    parser.add_argument("--explain",
                        action="store_true",
                        help="If true, a saved model is loaded and used in ranking segments to explain predictions")
    parser.add_argument("--eval_model_name",
                        default=None,
                        type=str,
                        help="The filename of the model to be loaded from the directory specified in --model_dir")

    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])
    setattr(flags, "cv", True if flags.fold_num > 2 else False)
    main(flags)
