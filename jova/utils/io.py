# Author: bbrighttaer
# Project: jova
# Date: 5/23/19
# Time: 10:43 AM
# File: io.py


from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import pickle

import deepchem
import padme

__author__ = 'Brighter Agyemang'

import os
import logging
import sys
import torch
import numpy as np


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
                                   prot_kernel_dict):
    assert fold_num > 1
    for i in range(fold_num):
        fold_dir = os.path.join(save_dir, "fold" + str(i + 1))
        train_dir = os.path.join(fold_dir, "train_dir")
        valid_dir = os.path.join(fold_dir, "valid_dir")
        test_dir = os.path.join(fold_dir, "test_dir")
        train_data = fold_dataset[i][0]
        valid_data = fold_dataset[i][1]
        test_data = fold_dataset[i][2]
        train_data.move(train_dir)
        valid_data.move(valid_dir)
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
    return None


def save_dataset_to_disk(save_dir, train, valid, test, transformers, gnn_fingerprint, drug_kernel_dict,
                         prot_kernel_dict):
    train_dir = os.path.join(save_dir, "train_dir")
    valid_dir = os.path.join(save_dir, "valid_dir")
    test_dir = os.path.join(save_dir, "test_dir")
    train.move(train_dir)
    valid.move(valid_dir)
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
        if not os.path.exists(train_dir) or not os.path.exists(valid_dir) or not os.path.exists(test_dir):
            return False, None, list(), None, None
        train = padme.data.DiskDataset(train_dir)
        valid = padme.data.DiskDataset(valid_dir)
        test = padme.data.DiskDataset(test_dir)
        train_data.append(train)
        valid_data.append(valid)
        test_data.append(test)

    gnn_fingerprint = None
    drug_sim_kernel_dict = prot_sim_kernel_dict = None

    if os.path.exists(os.path.join(save_dir, "gnn_fingerprint_dict.pkl")):
        with open(os.path.join(save_dir, "gnn_fingerprint_dict.pkl"), "rb") as f:
            gnn_fingerprint = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, "drug_drug_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "drug_drug_kernel_dict.pkl"), "rb") as f:
            drug_sim_kernel_dict = pickle.load(f)
    if os.path.exists(os.path.join(save_dir, "prot_prot_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "prot_prot_kernel_dict.pkl"), "rb") as f:
            prot_sim_kernel_dict = pickle.load(f)

    loaded = True
    with open(os.path.join(save_dir, "transformers.pkl"), 'rb') as f:
        transformers = pickle.load(f)
        return loaded, list(zip(train_data, valid_data, test_data)), transformers, gnn_fingerprint, \
               (drug_sim_kernel_dict, prot_sim_kernel_dict)


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
    if not os.path.exists(train_dir) or not os.path.exists(
            valid_dir) or not os.path.exists(test_dir):
        return False, None, list(), None, None

    gnn_fingerprint = None
    drug_sim_kernel_dict = prot_sim_kernel_dict = None

    if os.path.exists(os.path.join(save_dir, "gnn_fingerprint_dict.pkl")):
        with open(os.path.join(save_dir, "gnn_fingerprint_dict.pkl"), "rb") as f:
            gnn_fingerprint = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, "drug_drug_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "drug_drug_kernel_dict.pkl"), "rb") as f:
            drug_sim_kernel_dict = pickle.load(f)
    if os.path.exists(os.path.join(save_dir, "prot_prot_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "prot_prot_kernel_dict.pkl"), "rb") as f:
            prot_sim_kernel_dict = pickle.load(f)

    loaded = True
    train = deepchem.data.DiskDataset(train_dir)
    valid = deepchem.data.DiskDataset(valid_dir)
    test = deepchem.data.DiskDataset(test_dir)
    all_dataset = (train, valid, test)
    with open(os.path.join(save_dir, "transformers.pkl"), 'rb') as f:
        transformers = pickle.load(f)
        return loaded, all_dataset, transformers, gnn_fingerprint, (drug_sim_kernel_dict, prot_sim_kernel_dict)
