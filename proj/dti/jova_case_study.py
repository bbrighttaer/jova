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
import json
import os
import random
import time
from datetime import datetime as dt

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from soek import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import chain
import jova.metrics as mt
import jova.utils.io
from jova import cuda
from jova.data import batch_collator, get_data, load_proteins, DtiDataset
from jova.data.data import featurize_datasets
from jova.metrics import compute_model_performance
from jova.nn.layers import GraphConvLayer, GraphPool, Unsqueeze, GraphGather2D, ElementwiseBatchNorm
from jova.nn.models import GraphConvSequential, WeaveModel, NwayForward, JointAttention, Prot2Vec, ProteinRNN, \
    GraphNeuralNet2D, ProteinCNN2D, ProteinCNN
from jova.trans import undo_transforms
from jova.utils import Trainer
from jova.utils.args import WeaveLayerArgs, WeaveGatherArgs, Flags
from jova.utils.attn_helpers import AttentionDataService
from jova.utils.io import load_pickle
from jova.utils.train_helpers import count_parameters, FrozenModels, ViewsReg, parse_hparams

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1]

dvc_id = 0
torch.cuda.set_device(dvc_id)

joint_attention_data = AttentionDataService(True)

views_reg = ViewsReg()


def create_ecfp_net(hparams):
    model = nn.Sequential(Unsqueeze(dim=0))
    return model


def create_prot_net(hparams, model_type, protein_profile, frozen_models_hook=None):
    # assert protein_profile is not None, "Protein profile has to be supplied"
    # assert protein_embeddings is not None, "Pre-trained protein embeddings are required"

    # model_type = hparams["prot"]["model_type"].lower()
    assert (model_type in views_reg.all_prot_views), "Valid protein types: {}".format(
        str(views_reg.all_prot_views))
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


def create_integrated_net(hparams, protein_profile):
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
        joint_attention_data.labels.append(cview)

    # protein models
    for m_type in hparams["prot"]["model_types"]:
        views[m_type] = create_prot_net(hparams, m_type, protein_profile, frozen_models)
        if m_type == "psc":
            seg_dims.append(hparams["prot"]["psc_dim"])
        elif m_type == "rnn":
            seg_dims.append(hparams["prot"]["rnn_hidden_state_dim"])
        elif m_type == "p2v":
            seg_dims.append(hparams["prot"]["embedding_dim"] * hparams["prot"]["window"])
        elif m_type in ["pcnn", "pcnn2d"]:
            seg_dims.append(hparams["prot"]["embedding_dim"])

        # register the view of the protein for recording attention info
        joint_attention_data.labels.append(m_type)

    layers = [NwayForward(models=views.values())]

    func_callback = joint_attention_data.attn_forward_hook if hparams["explain_mode"] else None
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
    def initialize(hparams, train_dataset, val_dataset, test_dataset, protein_profile, cuda_devices=None,
                   mode="regression"):

        # create networks
        model, frozen_models = create_integrated_net(hparams, protein_profile)
        print("Number of trainable parameters: model={}".format(count_parameters(model)))
        if cuda:
            model = model.cuda()

        # data loaders
        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=1,
                                       shuffle=True,
                                       collate_fn=lambda x: x)
        val_data_loader = DataLoader(dataset=val_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     collate_fn=lambda x: x)
        if test_dataset is not None:
            test_data_loader = DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          collate_fn=lambda x: x)
        else:
            test_data_loader = None

        # metrics
        metrics = [mt.Metric(mt.rms_score, np.nanmean),
                   mt.Metric(mt.concordance_index, np.nanmean),
                   mt.Metric(mt.pearson_r2_score, np.nanmean)]
        return model, {"train": train_data_loader, "val": val_data_loader,
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
    def explain_model(model, model_dir, model_name, data_loaders, transformers_dict, prot_desc_dict, prot_model_types,
                      sim_data_node, k=10):
        # load saved model and put in evaluation mode
        model.load_state_dict(jova.utils.io.load_model(model_dir, model_name, dvc=torch.device(f"cuda:{dvc_id}")))
        model.eval()

        print("Model explanation...")
        start = time.time()

        # sub-nodes of sim data resource
        attn_ranking = []
        attn_ranking_node = DataNode(label="attn_ranking", data=attn_ranking)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [attn_ranking_node]

        # Since we're evaluating, join all data loaders
        all_loaders = chain()
        for loader in data_loaders:
            if data_loaders[loader] is not None:
                all_loaders = chain(all_loaders, data_loaders[loader])

        # Main evaluation loop
        for batch in tqdm(all_loaders):
            spec = {v: True for v in views_reg.c_views}
            batch_size, data = batch_collator(batch, prot_desc_dict, spec, cuda_prot=True)

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
                    if m_type in views_reg.embedding_based_views:
                        protein_xs.append(Xs[list(Xs.keys())[0]][2])
                    else:  # m_type == "psc":
                        protein_xs.append(Xs[list(Xs.keys())[0]][1])
                    attn_data_x[m_type] = Xs[list(Xs.keys())[0]][2]

                # compound data in batch
                X = [Xs[v][0] for v in views_reg.c_views]

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
                                'y_true': np_to_plot_data(y_true),
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


def main(pid, flags):
    # Load protein data
    prot_desc_dict, prot_seq_dict = load_proteins(flags['prot_desc_path'])
    prot_profile = load_pickle(file_name=flags.prot_profile)
    prot_vocab = load_pickle(file_name=flags.prot_vocab)
    flags["prot_vocab_size"] = len(prot_vocab)

    # set joint attention hook's protein information
    joint_attention_data.protein_profile = prot_profile
    joint_attention_data.protein_vocabulary = prot_vocab
    joint_attention_data.protein_sequences = prot_seq_dict

    # Ensures all possible compound data in each seed is present. Helps with maintaining the random number generator
    # state during splitting to avoid sample mismatch across views.
    featurize_datasets(flags.jova, views_reg.feat_dict, flags, prot_seq_dict, seeds)

    print("JOVA sims:", flags.jova)
    for v_arg in flags.jova:
        views_reg.parse_views(v_arg)
        comp_lbl = views_reg.c_views
        comp_lbl = '_'.join(comp_lbl)
        flags["prot_model_types"] = views_reg.p_views
        sim_label = "integrated_view_attn_no_gan_" + ('_'.join(flags["prot_model_types"])) + '_' + comp_lbl
        print("CUDA={}, view={}".format(cuda, sim_label))

        # Simulation data resource tree
        split_label = flags.split
        dataset_lbl = flags["dataset_name"]
        node_label = json.dumps({'model_family': 'jova',
                                 'dataset': dataset_lbl,
                                 'cviews': '-'.join(views_reg.c_views),
                                 'pviews': '-'.join(views_reg.p_views),
                                 'split': split_label,
                                 'mode': 'case_study',
                                 'model_ds': flags['eval_model_name'].split('_')[0],
                                 'target': flags['target'],
                                 'cv': str(flags['cv']),
                                 'seeds': '-'.join([str(s) for s in seeds]),
                                 'date': date_label})
        sim_data = DataNode(label=node_label)
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

            trainer = Jova()
            print(f'Case study label:{sim_label}')
            invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node,
                         sim_label, prot_profile)

        # save simulation data resource tree to file.
        sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node, view, protein_profile):
    try:
        hfile = os.path.join('soek_res', get_hparam_file())
        exists = os.path.exists(hfile)
        status = 'Found' if exists else 'Not Found, switching to default hyperparameters'
        print(f'Hyperparameters file:{hfile}, status={status}')
        if not exists:
            raise FileNotFoundError(f'{hfile} not found')
        hyper_params = parse_hparams(hfile)
        hyper_params['explain_mode'] = True
        hyper_params['gnn']['fingerprint_size'] = len(flags["gnn_fingerprint"]) \
            if flags["gnn_fingerprint"] is not None else 0
    except:
        hyper_params = default_hparams_bopt(flags)

    start_fold(data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer, transformers_dict, view,
               protein_profile)


def start_fold(sim_data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
               transformers_dict, view, protein_profile):
    data = trainer.data_provider(None, flags, data_dict)
    model, data_loaders, metrics, prot_model_types, frozen_models = trainer.initialize(hparams=hyper_params,
                                                                                       train_dataset=data["train"],
                                                                                       protein_profile=protein_profile,
                                                                                       val_dataset=data["val"],
                                                                                       test_dataset=data["test"])
    trainer.explain_model(model, flags["model_dir"], flags["eval_model_name"], data_loaders, transformers_dict,
                          prot_desc_dict, prot_model_types, sim_data_node)


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
        "explain_mode": True,
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
            "fingerprint_size": len(flags["gnn_fingerprint"]) if flags["gnn_fingerprint"] else 0,
            "num_layers": 3,
            "dim": 100,
        }
    }


def get_hparam_file():
    cv = '-'.join(views_reg.c_views)
    pv = '-'.join(views_reg.p_views)
    return {
        'ecfp8-gconv__psc': 'bayopt_search_integrated_view_attn_no_gan_psc_ecfp8_gconv_dti_2019_11_21__16_08_16_gp.csv',
        'ecfp8-gnn__psc': 'bayopt_search_integrated_view_attn_no_gan_psc_ecfp8_gnn_dti_2019_11_21__16_08_16_gp.csv',
        'ecfp8-gconv__pcnn2d': 'bayopt_search_integrated_view_attn_no_gan_pcnn2d_ecfp8_gconv_dti_2019_11_27__21_57_05_gp.csv',
        'ecfp8-weave__psc': 'bayopt_search_integrated_view_attn_no_gan_psc_ecfp8_weave_dti_2019_11_27__21_57_05_gp.csv',
        'ecfp8-gnn__pcnn2d-psc': 'bayopt_search_integrated_view_attn_no_gan_pcnn2d_psc_ecfp8_gnn_dti_2019_11_27__21_57_05_gp.csv',
        'ecfp8-gconv__rnn-psc': 'bayopt_search_integrated_view_attn_no_gan_rnn_psc_ecfp8_gconv_dti_2019_12_05__04_03_42_gbrt.csv',
    }.get(f'{cv}__{pv}', None)


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
    parser.add_argument("--splitting_alg",
                        choices=["random", "scaffold", "butina", "index", "task"],
                        default="random",
                        type=str,
                        help="Data splitting algorithm to use.")
    parser.add_argument('--filter_threshold',
                        type=int,
                        default=0,
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
    parser.add_argument('--mp', '-mp', action='store_true', help="Multiprocessing option")
    parser.add_argument('--target', type=str, help='The ID of the target/protein of the case study')

    args = parser.parse_args()
    procs = []
    use_mp = args.mp
    for split in args.split_schemes:
        flags = Flags()
        args_dict = args.__dict__
        for arg in args_dict:
            setattr(flags, arg, args_dict[arg])
        setattr(flags, "cv", False)
        flags['split'] = split
        flags['predict_cold'] = split == 'cold_drug_target'
        flags['cold_drug'] = split == 'cold_drug'
        flags['cold_target'] = split == 'cold_target'
        flags['cold_drug_cluster'] = split == 'cold_drug_cluster'
        flags['split_warm'] = split == 'warm'
        flags['test'] = False
        flags['fold_num'] = 0
        if use_mp:
            p = mp.spawn(fn=main, args=(flags,), join=False)
            procs.append(p)
        else:
            main(0, flags)
    for proc in procs:
        proc.join()
