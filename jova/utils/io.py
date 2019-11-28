# Author: bbrighttaer
# Project: jova
# Date: 5/23/19
# Time: 10:43 AM
# File: io.py


from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import pickle


__author__ = 'Brighter Agyemang'

import os
import logging
import sys
import torch
import numpy as np
import jova

def get_logger(name=None, level='INFO', stream='stderr', filename=None, log_dir='./logs/'):
    """
    Creates and return a logger to both console and a specified file.

    :param log_dir: The directory of the log file
    :param filename: The file to be logged into. It shall be in ./logs/
    :param name: The name of the logger
    :param level: The logging level; one of DEBUG, INFO, WARNING, ERROR, CRITICAL
    :return: The created logger
    :param stream: Either 'stderr' or 'stdout'
    """
    os.makedirs(log_dir, exist_ok=True)
    stream = sys.stderr if stream == 'stderr' else sys.stdout
    log_level = {'DEBUG': logging.DEBUG,
                 'INFO': logging.INFO,
                 'WARNING': logging.WARNING,
                 'ERROR': logging.ERROR,
                 'CRITICAL': logging.CRITICAL}.get(level.upper(), 'INFO')
    handlers = []
    if filename:
        handlers.append(logging.FileHandler(os.path.join(log_dir, filename + '.log')))
    if stream:
        handlers.append(logging.StreamHandler(stream))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=handlers)
    return logging.getLogger(name)


def save_model(model, path, name):
    """
    Saves the model parameters.

    :param model:
    :param path:
    :param name:
    :return:
    """
    os.makedirs(path, exist_ok=True)
    # file = os.path.join(path, name + ".mod")
    # torch.save(model.state_dict(), file)
    with open(os.path.join(path, "dummy_save.txt"), 'a') as f:
        f.write(name + '\n')


def save_dict_model(model, path, name):
    """
    Saves the model parameters.

    :param model:
    :param path:
    :param name:
    :return:
    """
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, name + ".pkl")
    # with open(file, 'wb') as f:
    #     pickle.dump(dict(model), f)
    with open(os.path.join(path, "dummy_save_dict.txt"), 'a') as f:
        f.write(name + '\n')


def load_dict_model(path, name):
    return load_pickle(os.path.join(path, name))


def load_model(path, name, dvc=torch.device("cuda:0")):
    """
    Loads the parameters of a model.

    :param path:
    :param name:
    :return: The saved state_dict.
    """
    return torch.load(os.path.join(path, name), map_location=dvc)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def load_numpy_array(file_name):
    return np.load(file_name, allow_pickle=True)


def save_nested_cv_dataset_to_disk(save_dir, fold_dataset, fold_num, transformers, gnn_fingerprint, drug_kernel_dict,
                                   prot_kernel_dict, simboost_pairwise_feats_dict):
    assert fold_num > 1
    for i in range(fold_num):
        fold_dir = os.path.join(save_dir, "fold" + str(i + 1))
        train_dir = os.path.join(fold_dir, "train_dir")
        valid_dir = os.path.join(fold_dir, "valid_dir")
        test_dir = os.path.join(fold_dir, "test_dir")
        train_data = fold_dataset[i][0]
        valid_data = fold_dataset[i][1]
        test_data = fold_dataset[i][2]
        if train_data:
            train_data.move(train_dir)
        if valid_data:
            valid_data.move(valid_dir)
        if test_data:
            test_data.move(test_dir)
    with open(os.path.join(save_dir, "transformers.pkl"), "wb") as f:
        pickle.dump(transformers, f)
    if gnn_fingerprint is not None:
        with open(os.path.join(save_dir, "gnn_fingerprint_dict.pkl"), "wb") as f:
            pickle.dump(dict(gnn_fingerprint), f)
    if drug_kernel_dict is not None:
        with open(os.path.join(save_dir, "drug_drug_kernel_dict.pkl"), "wb") as f:
            pickle.dump(dict(drug_kernel_dict), f)
    if prot_kernel_dict is not None:
        with open(os.path.join(save_dir, "prot_prot_kernel_dict.pkl"), "wb") as f:
            pickle.dump(dict(prot_kernel_dict), f)
    if simboost_pairwise_feats_dict is not None:
        with open(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl"), "wb") as f:
            pickle.dump(dict(simboost_pairwise_feats_dict), f)
    return None


def save_dataset_to_disk(save_dir, train, valid, test, transformers, gnn_fingerprint, drug_kernel_dict,
                         prot_kernel_dict, simboost_pairwise_feats_dict):
    train_dir = os.path.join(save_dir, "train_dir")
    valid_dir = os.path.join(save_dir, "valid_dir")
    test_dir = os.path.join(save_dir, "test_dir")
    if train:
        train.move(train_dir)
    if valid:
        valid.move(valid_dir)
    if test:
        test.move(test_dir)
    with open(os.path.join(save_dir, "transformers.pkl"), 'wb') as f:
        pickle.dump(transformers, f)
    if gnn_fingerprint is not None:
        with open(os.path.join(save_dir, "gnn_fingerprint_dict.pkl"), "wb") as f:
            pickle.dump(gnn_fingerprint, f)
    if drug_kernel_dict is not None:
        with open(os.path.join(save_dir, "drug_drug_kernel_dict.pkl"), "wb") as f:
            pickle.dump(dict(drug_kernel_dict), f)
    if prot_kernel_dict is not None:
        with open(os.path.join(save_dir, "prot_prot_kernel_dict.pkl"), "wb") as f:
            pickle.dump(dict(prot_kernel_dict), f)
    if simboost_pairwise_feats_dict is not None:
        with open(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl"), "wb") as f:
            pickle.dump(dict(simboost_pairwise_feats_dict), f)
    return None


def load_nested_cv_dataset_from_disk(save_dir, fold_num):
    assert fold_num > 1
    loaded = False
    train_data = []
    valid_data = []
    test_data = []
    for i in range(fold_num):
        fold_dir = os.path.join(save_dir, "fold" + str(i + 1))
        train_dir = os.path.join(fold_dir, "train_dir")
        valid_dir = os.path.join(fold_dir, "valid_dir")
        test_dir = os.path.join(fold_dir, "test_dir")
        if not os.path.exists(train_dir):
            return False, None, list(), None, None, None
        train = jova.data.DiskDataset(train_dir)
        valid = jova.data.DiskDataset(valid_dir) if os.path.exists(valid_dir) else None
        test = jova.data.DiskDataset(test_dir) if os.path.exists(test_dir) else None
        train_data.append(train)
        valid_data.append(valid)
        test_data.append(test)

    gnn_fingerprint = None
    simboost_pairwise_feats_dict = drug_sim_kernel_dict = prot_sim_kernel_dict = None

    if os.path.exists(os.path.join(save_dir, "gnn_fingerprint_dict.pkl")):
        with open(os.path.join(save_dir, "gnn_fingerprint_dict.pkl"), "rb") as f:
            gnn_fingerprint = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, "drug_drug_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "drug_drug_kernel_dict.pkl"), "rb") as f:
            drug_sim_kernel_dict = pickle.load(f)
    if os.path.exists(os.path.join(save_dir, "prot_prot_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "prot_prot_kernel_dict.pkl"), "rb") as f:
            prot_sim_kernel_dict = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl")):
        with open(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl"), "rb") as f:
            simboost_pairwise_feats_dict = pickle.load(f)

    loaded = True
    with open(os.path.join(save_dir, "transformers.pkl"), 'rb') as f:
        transformers = pickle.load(f)
        return loaded, list(zip(train_data, valid_data, test_data)), transformers, gnn_fingerprint, \
               (drug_sim_kernel_dict, prot_sim_kernel_dict), simboost_pairwise_feats_dict


def load_dataset_from_disk(save_dir):
    """
    Parameters
    ----------
    save_dir: str

    Returns
    -------
    loaded: bool
      Whether the load succeeded
    all_dataset: (dc.data.Dataset, dc.data.Dataset, dc.data.Dataset)
      The train, valid, test datasets
    transformers: list of dc.trans.Transformer
      The transformers used for this dataset

    """

    train_dir = os.path.join(save_dir, "train_dir")
    valid_dir = os.path.join(save_dir, "valid_dir")
    test_dir = os.path.join(save_dir, "test_dir")
    if not os.path.exists(train_dir):
        return False, None, list(), None, None, None

    gnn_fingerprint = None
    simboost_pairwise_feats_dict = drug_sim_kernel_dict = prot_sim_kernel_dict = None

    if os.path.exists(os.path.join(save_dir, "gnn_fingerprint_dict.pkl")):
        with open(os.path.join(save_dir, "gnn_fingerprint_dict.pkl"), "rb") as f:
            gnn_fingerprint = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, "drug_drug_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "drug_drug_kernel_dict.pkl"), "rb") as f:
            drug_sim_kernel_dict = pickle.load(f)
    if os.path.exists(os.path.join(save_dir, "prot_prot_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "prot_prot_kernel_dict.pkl"), "rb") as f:
            prot_sim_kernel_dict = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl")):
        with open(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl"), "rb") as f:
            simboost_pairwise_feats_dict = pickle.load(f)

    loaded = True
    train = jova.data.DiskDataset(train_dir)
    valid = jova.data.DiskDataset(valid_dir) if os.path.exists(valid_dir) else None
    test = jova.data.DiskDataset(test_dir) if os.path.exists(test_dir) else None
    all_dataset = (train, valid, test)
    with open(os.path.join(save_dir, "transformers.pkl"), 'rb') as f:
        transformers = pickle.load(f)
        return loaded, all_dataset, transformers, gnn_fingerprint, \
               (drug_sim_kernel_dict, prot_sim_kernel_dict), simboost_pairwise_feats_dict
