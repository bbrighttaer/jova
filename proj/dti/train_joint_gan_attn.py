# Author: bbrighttaer
# Project: jova
# Date: 7/22/19
# Time: 10:47 AM
# File: train_joint.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import json
import logging
import random
import time
from datetime import datetime as dt

import numpy as np
import torch
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
from jova.data.data import featurize_datasets
from jova.metrics import compute_model_performance
from jova.nn.layers import GraphConvLayer, GraphPool, GraphGather2D, Unsqueeze, ElementwiseBatchNorm
from jova.nn.models import GraphConvSequential, create_fcn_layers, NwayForward, JointAttention, \
    WeaveModel, Prot2Vec, ProteinRNN, ProteinCNN, ProteinCNN2D, GraphNeuralNet2D
from jova.trans import undo_transforms
from jova.utils import Trainer
from jova.utils.args import FcnArgs, WeaveGatherArgs, WeaveLayerArgs
from jova.utils.io import load_pickle
from jova.utils.math import ExpAverage, Count
from jova.utils.tb import TBMeanTracker
from jova.utils.train_helpers import count_parameters, GradStats, FrozenModels, ViewsReg

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.ERROR, filename='error.log')

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1, 8, 64]

check_data = False

torch.cuda.set_device(2)

views_reg = ViewsReg()


def create_ecfp_net(hparams):
    model = nn.Sequential(Unsqueeze(dim=0))
    return model


def create_prot_net(hparams, model_type, protein_profile, protein_embeddings, frozen_models_hook=None):
    # assert protein_profile is not None, "Protein profile has to be supplied"
    # assert protein_embeddings is not None, "Pre-trained protein embeddings are required"

    # model_type = hparams["prot"]["model_type"].lower()
    assert (model_type in ViewsReg.all_prot_views), "Valid protein types: {}".format(
        str(ViewsReg.all_prot_views))
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

    net_creators = {'ecfp8': create_ecfp_net,
                    'weave': create_weave_net,
                    'gconv': create_gconv_net,
                    'gnn': create_gnn_net}

    # compound models
    for cview in views_reg.c_views:
        views[cview] = net_creators[cview](hparams)
        seg_dims.append(hparams[cview]["dim"])

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

    layers = [NwayForward(models=views.values())]

    layers.append(JointAttention(d_dims=seg_dims, latent_dim=hparams["latent_dim"], num_heads=hparams["attn_heads"],
                                 num_layers=hparams["attn_layers"], dprob=hparams["dprob"]))

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


def create_discriminator_net(hparams):
    fcn_args = []
    p = hparams["neigh_dist"]
    layers = hparams["disc_hdims"]
    if not isinstance(layers, list):
        layers = [layers]
    for dim in layers:
        conf = FcnArgs(in_features=p,
                       out_features=dim,
                       activation='relu',
                       batch_norm=True,
                       dropout=hparams["dprob"])
        fcn_args.append(conf)
        p = dim
    fcn_args.append(FcnArgs(in_features=p, out_features=1, activation="sigmoid"))
    layers = create_fcn_layers(fcn_args)
    model = nn.Sequential(*layers)
    return model


class JovaGAN(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset, protein_profile, protein_embeddings,
                   cuda_devices=None, mode="regression"):

        # create networks
        generator, frozen_models = create_integrated_net(hparams, protein_profile, protein_embeddings)
        discriminator = create_discriminator_net(hparams)
        print("Number of trainable parameters: generator={}, discriminator={}".format(count_parameters(generator),
                                                                                      count_parameters(discriminator)))

        try:
            if cuda:
                generator = generator.cuda()
                discriminator = discriminator.cuda()
        except:  # todo: Remove this try-catch after hyperparameter search
            generator = nn.Sequential(nn.Linear(10, 10))
            discriminator = nn.Sequential(nn.Linear(10, 10))

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

        # filter optimizer arguments
        optimizer_disc = optimizer_gen = None
        for suffix in ["_gen", "_disc"]:
            key = "optimizer{}".format(suffix)

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
            }.get(hparams[key].lower(), None)
            assert optimizer is not None, "{} optimizer could not be found"

            optim_kwargs = dict()
            optim_key = hparams[key]
            for k, v in hparams.items():
                if "optimizer{}__".format(suffix) in k:
                    attribute_tup = k.split("__")
                    if optim_key == attribute_tup[1] or attribute_tup[1] == "global":
                        optim_kwargs[attribute_tup[2]] = v
            if suffix == "_gen":
                optimizer_gen = optimizer(generator.parameters(), **optim_kwargs)
            else:
                optimizer_disc = optimizer(discriminator.parameters(), **optim_kwargs)

        # metrics
        metrics = [mt.Metric(mt.rms_score, np.nanmean),
                   mt.Metric(mt.concordance_index, np.nanmean),
                   mt.Metric(mt.pearson_r2_score, np.nanmean)]
        return (generator, discriminator), (optimizer_gen, optimizer_disc), \
               {"train": train_data_loader,
                "val": val_data_loader,
                "test": test_data_loader}, metrics, hparams["prot"]["model_types"], \
               hparams["weighted_loss"], hparams["neigh_dist"], frozen_models

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
    def train(models, optimizers, data_loaders, metrics, prot_model_types, weighted_loss, neigh_dist,
              frozen_models, transformers_dict, prot_desc_dict, tasks, n_iters=5000, sim_data_node=None,
              epoch_ckpt=(2, 1.0), tb_writer=None, is_hsearch=False):
        generator, discriminator = models
        optimizer_gen, optimizer_disc = optimizers

        tb_writer = tb_writer()
        start = time.time()
        best_model_wts = generator.state_dict()
        best_score = -10000
        best_epoch = -1
        terminate_training = False
        e_avg = ExpAverage(.01)
        n_epochs = n_iters // len(data_loaders["train"])
        grad_stats = GradStats(nn.ModuleList(models), beta=0.)

        # learning rate decay schedulers
        scheduler_gen = sch.StepLR(optimizer_gen, step_size=500, gamma=0.01)
        scheduler_disc = sch.StepLR(optimizer_disc, step_size=500, gamma=0.01)

        # pred_loss functions
        prediction_criterion = nn.MSELoss()
        adversarial_loss = nn.BCELoss()

        # sub-nodes of sim data resource
        loss_lst = []
        train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        val_scores_lst = []
        val_scores_node = DataNode(label="validation_score", data=val_scores_lst)
        train_scores_lst = []
        train_scores_node = DataNode(label="train_score", data=train_scores_lst)
        gen_loss_lst = []
        gen_loss_node = DataNode(label="generator_loss", data=gen_loss_lst)
        dis_loss_lst = []
        dis_loss_node = DataNode(label="discriminator_loss", data=dis_loss_lst)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [train_loss_node, train_scores_node, metrics_node, val_scores_node, gen_loss_node,
                                  dis_loss_node]

        try:
            # Main training loop
            tb_idx = {'train': Count(), 'val': Count(), 'test': Count()}
            for epoch in range(n_epochs):
                if terminate_training:
                    print("Terminating training...")
                    break
                for phase in ["train", "val" if is_hsearch else "test"]:
                    # ensure these models are frozen at all times
                    for m in frozen_models:
                        m.eval()

                    if phase == "train":
                        print("Training....")
                        # Training mode
                        generator.train()
                        discriminator.train()
                    else:
                        print("Validation...")
                        # Evaluation mode
                        generator.eval()

                    data_size = 0.
                    epoch_losses = []
                    epoch_scores = []

                    # Iterate through mini-batches
                    i = 0
                    with TBMeanTracker(tb_writer, 10) as tracker:
                        with grad_stats:
                            for batch in tqdm(data_loaders[phase]):
                                spec = {v: True for v in views_reg.c_views}
                                batch_size, data = batch_collator(batch, prot_desc_dict, spec, cuda_prot=True)

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

                                optimizer_gen.zero_grad()
                                optimizer_disc.zero_grad()

                                # forward propagation
                                # track history if only in train
                                with torch.set_grad_enabled(phase == "train"):
                                    Ys = {k: Ys[k] for k in Ys}
                                    # Ensure corresponding pairs
                                    for j in range(1, len(Ys.values())):
                                        assert (list(Ys.values())[j - 1] == list(Ys.values())[j]).all()

                                    y = Ys[list(Xs.keys())[0]]
                                    w = Ws[list(Xs.keys())[0]]

                                    # protein data in batch
                                    protein_xs = []
                                    for m_type in prot_model_types:
                                        if m_type in views_reg.embedding_based_views:
                                            protein_xs.append(Xs[list(Xs.keys())[0]][2])
                                        else:  # m_type == "psc":
                                            protein_xs.append(Xs[list(Xs.keys())[0]][1])

                                    # compound data in batch
                                    X = [Xs[v][0] for v in views_reg.c_views]

                                    # merge compound and protein list
                                    X = X + protein_xs

                                    outputs = generator(X)
                                    target = torch.from_numpy(y).float()
                                    weights = torch.from_numpy(w).float()
                                    valid = torch.ones_like(target).float()
                                    fake = torch.zeros_like(target).float()
                                    if cuda:
                                        target = target.cuda()
                                        weights = weights.cuda()
                                        valid = valid.cuda()
                                        fake = fake.cuda()
                                    outputs = outputs * weights
                                    pred_loss = prediction_criterion(outputs, target)

                                # fail fast
                                if str(pred_loss.item()) == "nan":
                                    terminate_training = True
                                    break

                                eval_dict = {}
                                score = JovaGAN.evaluate(eval_dict, y, outputs, w, metrics, tasks,
                                                         transformers_dict[list(Xs.keys())[0]])
                                # TBoard info
                                tracker.track("%s/loss" % phase, pred_loss.item(), tb_idx[phase].IncAndGet())
                                tracker.track("%s/score" % phase, score, tb_idx[phase].i)
                                for k in eval_dict:
                                    tracker.track('{}/{}'.format(phase, k), eval_dict[k], tb_idx[phase].i)

                                if phase == "train":
                                    # GAN stuff
                                    f_xx, f_yy = torch.meshgrid(outputs.squeeze(), outputs.squeeze())
                                    predicted_diffs = torch.abs(f_xx - f_yy).sort(dim=1)[0][:, : neigh_dist]
                                    r_xx, r_yy = torch.meshgrid(target.squeeze(), target.squeeze())
                                    real_diffs = torch.abs(r_xx - r_yy).sort(dim=1)[0][:, :neigh_dist]

                                    # generator
                                    gen_loss = adversarial_loss(discriminator(predicted_diffs), valid)
                                    gen_loss_lst.append(gen_loss.item())
                                    loss = pred_loss + weighted_loss * gen_loss
                                    tracker.track("gan/gen_loss", gen_loss.item(), tb_idx[phase].i)
                                    tracker.track("train/comp_loss", loss.item(), tb_idx[phase].i)
                                    loss.backward()
                                    optimizer_gen.step()

                                    # discriminator
                                    true_loss = adversarial_loss(discriminator(real_diffs), valid)
                                    fake_loss = adversarial_loss(discriminator(predicted_diffs.detach()), fake)
                                    discriminator_loss = (true_loss + fake_loss) / 2.
                                    tracker.track("gan/disc_loss", discriminator_loss.item(), tb_idx[phase].i)
                                    dis_loss_lst.append(discriminator_loss.item())
                                    discriminator_loss.backward()
                                    optimizer_disc.step()

                                    # for epoch stats
                                    epoch_losses.append(pred_loss.item())

                                    # for sim data resource
                                    loss_lst.append(pred_loss.item())

                                    print("\tEpoch={}/{}, batch={}/{}, pred_loss={:.4f}, D loss={:.4f}, "
                                          "G loss={:.4f}".format(epoch + 1, n_epochs, i + 1, len(data_loaders[phase]),
                                                                 pred_loss.item(), discriminator_loss, gen_loss))

                                    # monitor train data score
                                    train_scores_lst.append(score)
                                else:
                                    # for epoch stats
                                    epoch_scores.append(score)

                                    # for sim data resource
                                    val_scores_lst.append(score)
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
                        print("\nPhase: {}, avg task pred_loss={:.4f}, ".format(phase, np.nanmean(epoch_losses)))

                        # Adjust the learning rate.
                        scheduler_gen.step()
                        scheduler_disc.step()
                    else:
                        mean_score = np.mean(epoch_scores)
                        if best_score < mean_score:
                            best_score = mean_score
                            best_model_wts = copy.deepcopy(generator.state_dict())
                            best_epoch = epoch
        except RuntimeError as e:
            print(str(e))

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        try:
            generator.load_state_dict(best_model_wts)
        except RuntimeError as e:
            print(str(e))
        return {'model': generator, 'score': best_score, 'epoch': best_epoch}

    @staticmethod
    def evaluate_model(model, model_dir, model_name, data_loaders, metrics, prot_model_types,
                       transformers_dict, prot_desc_dict, tasks, sim_data_node=None):
        # load saved model and put in evaluation mode
        model.load_state_dict(jova.utils.io.load_model(model_dir, model_name))
        model.eval()

        print("Model evaluation...")
        start = time.time()
        n_epochs = 1

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

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [metrics_node, scores_node, model_preds_node]

        # Main evaluation loop
        for epoch in range(n_epochs):

            for phase in ["test"]:
                # Iterate through mini-batches
                i = 0
                for batch in tqdm(data_loaders[phase]):
                    spec = {v: True for v in views_reg.c_views}
                    batch_size, data = batch_collator(batch, prot_desc_dict, spec, cuda_prot=True)

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

                    # forward propagation
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
                            if m_type in views_reg.embedding_based_views:
                                protein_xs.append(Xs[list(Xs.keys())[0]][2])
                            else:  # m_type == "psc":
                                protein_xs.append(Xs[list(Xs.keys())[0]][1])

                        # compound data in batch
                        X = [Xs[v][0] for v in views_reg.c_views]

                        # merge compound and protein list
                        X = X + protein_xs
                        y_predicted = model(X)

                        # apply transformers
                        predicted_vals.extend(undo_transforms(y_predicted.cpu().detach().numpy(),
                                                              transformers_dict["gconv"]).squeeze().tolist())
                        true_vals.extend(
                            undo_transforms(y_true, transformers_dict["gconv"]).astype(np.float).squeeze().tolist())

                    eval_dict = {}
                    score = JovaGAN.evaluate(eval_dict, y_true, y_predicted, w, metrics, tasks,
                                             transformers_dict["gconv"])

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
    print("JOVA sims:", flags.jova)

    # Load protein data
    prot_desc_dict, prot_seq_dict = load_proteins(flags['prot_desc_path'])
    prot_profile = load_pickle(file_name=flags.prot_profile)
    prot_vocab = load_pickle(file_name=flags.prot_vocab)
    pretrained_embeddings = None  # load_numpy_array(flags.protein_embeddings)
    flags["prot_vocab_size"] = len(prot_vocab)

    # Ensures all possible compound data in each seed is present. Helps with maintaining the random number generator
    # state during splitting to avoid sample mismatch across views.
    featurize_datasets(flags.jova, views_reg.feat_dict, flags, prot_seq_dict, seeds)
    for v_arg in flags.jova:
        views_reg.parse_views(v_arg)
        comp_lbl = views_reg.c_views
        comp_lbl = '_'.join(comp_lbl)
        flags["prot_model_types"] = views_reg.p_views
        sim_label = "integrated_view_attn_gan_" + ('_'.join(flags["prot_model_types"])) + '_' + comp_lbl
        print("CUDA={}, view={}".format(cuda, sim_label))

        # Simulation data resource tree
        split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
            flags["cold_drug"] else "None"
        dataset_lbl = flags["dataset_name"]
        # node_label = "{}_{}_{}_{}_{}".format(dataset_lbl, sim_label, split_label, "eval" if flags["eval"] else "train",
        #                                      date_label)
        node_label = json.dumps({'model_family': 'jova-gan',
                                 'dataset': dataset_lbl,
                                 'cviews': '-'.join(views_reg.c_views),
                                 'pviews': '-'.join(views_reg.p_views),
                                 'split': split_label,
                                 'mode': "eval" if flags["eval"] else "train",
                                 'seeds': '-'.join([str(s) for s in seeds]),
                                 'date': date_label})
        sim_data = DataNode(label=node_label)
        nodes_list = []
        sim_data.data = nodes_list

        num_cuda_dvcs = torch.cuda.device_count()
        cuda_devices = None if num_cuda_dvcs == 1 else [i for i in range(1, num_cuda_dvcs)]

        # For searching over multiple seeds
        hparam_search = None

        for seed in seeds:
            summary_writer_creator = lambda: SummaryWriter(log_dir="tb_jovagan"
                                                                   "/{}_{}_{}/".format(sim_label, seed,
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
            for cview in views_reg.c_views:
                data_dict[cview] = get_data(views_reg.feat_dict[cview], flags, prot_sequences=prot_seq_dict, seed=seed)
                transformers_dict[cview] = data_dict[cview][2]
                flags["gnn_fingerprint"] = data_dict[cview][3]

            tasks = data_dict[list(data_dict.keys())[0]][0]
            # multi-task or single task is determined by the number of tasks w.r.t. the dataset loaded
            flags["tasks"] = tasks

            trainer = JovaGAN()

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
                    total_iterations = 3000
                    extra_train_args = {"transformers_dict": transformers_dict,
                                        "prot_desc_dict": prot_desc_dict,
                                        "tasks": tasks,
                                        "is_hsearch": True,
                                        "n_iters": total_iterations,
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
                                                   save_model_fn=jova.utils.io.save_model,
                                                   init_args=extra_init_args,
                                                   data_args=extra_data_args,
                                                   train_args=extra_train_args,
                                                   data_node=data_node,
                                                   split_label=split_label,
                                                   sim_label=sim_label,
                                                   minimizer=min_opt,
                                                   dataset_label=dataset_lbl,
                                                   results_file="{}_{}_dti_{}_{}-{}.csv".format(
                                                       flags["hparam_search_alg"], sim_label, date_label,
                                                       total_iterations, min_opt))

                    stats = hparam_search.fit(model_dir="models", model_name="".join(tasks), max_iter=20, seed=seed)
                    print(stats)
                    print("Best params = {}".format(stats.best()))
                else:
                    invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node,
                                 sim_label,
                                 prot_profile, pretrained_embeddings, summary_writer_creator)

        # save simulation data resource tree to file.
        sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node, view, protein_profile,
                 protein_embeddings, tb_writer):
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
                       transformers_dict, view, protein_profile, protein_embeddings, tb_writer, k)
    else:
        start_fold(data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
                   transformers_dict, view, protein_profile, protein_embeddings, tb_writer)


def start_fold(sim_data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
               transformers_dict, view, protein_profile, protein_embeddings, tb_writer=None, k=None):
    data = trainer.data_provider(k, flags, data_dict)
    model, optimizer, data_loaders, metrics, prot_model_types, weighted_loss, \
    neigh_dist, frozen_models = trainer.initialize(hparams=hyper_params,
                                                   train_dataset=data["train"],
                                                   val_dataset=data["val"],
                                                   test_dataset=data["test"],
                                                   protein_profile=protein_profile,
                                                   protein_embeddings=protein_embeddings)
    if flags["eval"]:
        trainer.evaluate_model(model[0], flags["model_dir"], flags["eval_model_name"],
                               data_loaders, metrics, transformers_dict,
                               prot_desc_dict, tasks, sim_data_node=sim_data_node)
    else:
        # Train the model
        results = trainer.train(model, optimizer, data_loaders, metrics, prot_model_types,
                                weighted_loss, neigh_dist, frozen_models, transformers_dict, prot_desc_dict,
                                tasks, max_iter=10000, sim_data_node=sim_data_node, tb_writer=tb_writer)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
            flags["cold_drug"] else "None"
        jova.utils.io.save_model(model, flags["model_dir"],
                                 "{}_{}_{}_{}_{}_{:.4f}".format(flags["dataset"], view, flags["model_name"],
                                                                split_label, epoch,
                                                                score))


def default_hparams_rand(flags):
    return {
        "prot_dim": 8421,
        "fp_dim": 1024,
        "gconv_dim": 128,
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
        "prot_vocab_size": flags["prot_vocab_size"],
        "attn_heads": 4,
        "attn_layers": 1,
        "lin_dims": [1424, 64],
        "output_dim": len(flags.tasks),
        "latent_dim": 256,
        "disc_hdims": [365, ],

        # weight initialization
        "kaiming_constant": 5,

        "weighted_loss": 0.1,

        # dropout
        "dprob": 0.1,

        "neigh_dist": 56,

        "tr_batch_size": 256,
        "val_batch_size": 128,
        "test_batch_size": 128,

        # optimizer params
        "optimizer_gen": "adamax",
        "optimizer_gen__global__weight_decay": 0.0009,
        "optimizer_gen__global__lr": 0.0006,
        "optimizer_disc": "sgd",
        "optimizer_disc__global__weight_decay": 1.0,
        "optimizer_disc__global__lr": 0.0001,
        "prot": {
            "model_types": flags["prot_model_types"],
            "vocab_size": flags["prot_vocab_size"],
            "window": 11,
            "pcnn_num_layers": 2,
            "embedding_dim": 10,
            "psc_dim": 8421,
            "rnn_hidden_state_dim": 10
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
            "fingerprint_size": len(flags["gnn_fingerprint"]) if flags["gnn_fingerprint"] else 0,
            "num_layers": 3,
            "dim": 100,
        }
    }


def get_hparam_config(flags):
    config = {
        "attn_heads": CategoricalParam([1, 2, 4, 8, 16]),
        "attn_layers": DiscreteParam(min=1, max=2),
        "lin_dims": DiscreteParam(min=64, max=1048, size=DiscreteParam(min=1, max=2)),
        "latent_dim": ConstantParam(256),
        "output_dim": ConstantParam(len(flags.tasks)),
        "disc_hdims": DiscreteParam(min=100, max=1048, size=2),

        # weight initialization
        "kaiming_constant": ConstantParam(5),

        "weighted_loss": RealParam(0.1, max=0.5),

        # dropout
        "dprob": RealParam(0.1, max=0.5),
        "neigh_dist": DiscreteParam(min=4, max=64),

        "tr_batch_size": ConstantParam(256),
        "val_batch_size": ConstantParam(128),
        "test_batch_size": ConstantParam(128),

        # optimizer params
        "optimizer_gen": CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
        "optimizer_gen__global__weight_decay": LogRealParam(),
        "optimizer_gen__global__lr": LogRealParam(),
        "optimizer_disc": CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
        "optimizer_disc__global__weight_decay": LogRealParam(),
        "optimizer_disc__global__lr": LogRealParam(),

        "prot": DictParam({
            "model_types": ConstantParam(flags["prot_model_types"]),
            "vocab_size": ConstantParam(flags["prot_vocab_size"]),
            "window": ConstantParam(11),
            "psc_dim": ConstantParam(8421)
        }),
        "ecfp8": DictParam({"dim": ConstantParam(1024)}),
        "weave": DictParam({}),
        "gconv": DictParam({}),
        "gnn": DictParam({})
    }

    # prot additions
    for pv in views_reg.p_views:
        # common props
        if pv in views_reg.embedding_based_views:
            config["prot"]["embedding_dim"] = DiscreteParam(min=5, max=50)

        # view-specific
        if pv in views_reg.pcnn_views:
            config["prot"]["pcnn_num_layers"] = DiscreteParam(min=1, max=4)
        elif pv == 'rnn':
            config["prot"]["rnn_hidden_state_dim"] = DiscreteParam(min=5, max=50)

    # comp additions
    for cv in views_reg.c_views:
        # common prop
        if cv in views_reg.graph_based_views:
            config[cv]["dim"] = DiscreteParam(min=64, max=512)

        # view-specific
        if cv == "gnn":
            config[cv]["fingerprint_size"] = ConstantParam(len(flags["gnn_fingerprint"]))
            config[cv]["num_layers"] = DiscreteParam(1, 4)
        elif cv == "weave":
            config[cv]["update_pairs"] = ConstantParam(False)
    return config


def verify_multiview_data(data_dict):
    ecfp8_data = data_dict["ecfp8"][1][0]
    gconv_data = data_dict["gconv"][1][0]
    corr = []
    for i in range(100):
        print("-" * 100)
        ecfp8 = "mol={}, prot={}, y={}".format(ecfp8_data.X[i][0].smiles, ecfp8_data.X[i][1].get_name(),
                                               ecfp8_data.y[i])
        print("ecfp8:", ecfp8)
        gconv = "mol={}, prot={}, y={}".format(gconv_data.X[i][0].smiles, gconv_data.X[i][1].get_name(),
                                               gconv_data.y[i])
        print("gconv:", gconv)
        print('#' * 100)
        corr.append(ecfp8 == gconv)
    print(corr)


class Flags(object):
    # enables using either object referencing or dict indexing to retrieve user passed arguments of flag objects.
    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        setattr(self, key, value)


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
    # parser.add_argument('--data_dir',
    #                     type=str,
    #                     default='../../data/',
    #                     help='Root folder of data (Davis, KIBA, Metz) folders.')
    parser.add_argument("--hparam_search",
                        action="store_true",
                        help="If true, hyperparameter searching would be performed.")
    parser.add_argument("--hparam_search_alg",
                        type=str,
                        default="bayopt_search",
                        help="Hyperparameter search algorithm to use. One of [bayopt_search, random_search]")
    parser.add_argument("--eval",
                        action="store_true",
                        help="If true, a saved model is loaded and evaluated using CV")
    parser.add_argument("--eval_model_name",
                        default=None,
                        type=str,
                        help="The filename of the model to be loaded from the directory specified in --model_dir")
    parser.add_argument("--jova",
                        action="append",
                        type=str,
                        help="A combination of compound and protein views for simulation. "
                             "The format is: comp1-compN;prot1-protN\nor instance, for a combination "
                             "of the PSC and RNN protein views with ECFP8 and GraphConv views of a compound, "
                             "the argument would be:\tecfp8-gconv;psc-rnn\n"
                             "Available compound views:[ecfp8,weave,gconv,gnn]\n Available protien views:"
                             "[psc,rnn,p2v,pcnn, pcnn2d]")

    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])
    setattr(flags, "cv", True if flags.fold_num > 2 else False)
    main(flags)
