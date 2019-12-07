from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import pandas as pd

import jova
import jova.splits as splits
from jova.feat import GNNFeaturizer
from jova.utils.io import save_nested_cv_dataset_to_disk, save_dataset_to_disk, load_nested_cv_dataset_from_disk, \
    load_dataset_from_disk


def load_csv_dataset(dataset_name, dataset_file, featurizer='Weave', cross_validation=False, test=False, split='random',
                     reload=True, K=5, mode='regression', predict_cold=False, cold_drug=False, cold_target=False,
                     cold_drug_cluster=False, split_warm=False, filter_threshold=0, prot_seq_dict=None,
                     oversampled=False, input_protein=True, seed=0, gnn_radius=2, simboost_mf_feats_dict=None):
    if cross_validation:
        assert not test

    data_dir, file_name = os.path.split(dataset_file)

    feat_label = featurizer
    gnn_fingerprint = None
    MF_entities_dict = None
    # for SimBoost, Kron-RLS and other kernel-based methods
    simboost_drug_target_feats_dict = drug_sim_kernel_dict = prot_sim_kernel_dict = None

    if input_protein:
        t_suffix = ''
        if mode == 'classification':
            t_suffix = '_bin'
        # headers = list(pd.read_csv(os.path.join(data_dir, file_name), header=0, nrows=0))
        tasks = [dataset_name + t_suffix]
    else:
        dataset_file = os.path.join(data_dir, file_name)
        headers = list(pd.read_csv(dataset_file, header=0, index_col=False, nrows=0))
        tasks = headers[:-1]

    if reload:
        delim = "/"
        if not input_protein:
            delim = "_no_prot" + delim
        if filter_threshold > 0:
            delim = "_filtered" + delim
        if predict_cold:
            delim = "_cold" + delim
        elif split_warm:
            delim = "_warm" + delim
        elif cold_drug:
            delim = "_cold_drug" + delim
        elif cold_target:
            delim = "_cold_target" + delim
        elif cold_drug_cluster:
            delim = '_cold_drug_cluster' + delim
        if oversampled:
            delim = "_oversp" + delim
        if cross_validation:
            delim = "_CV" + delim
            save_dir = os.path.join(data_dir, featurizer + delim + mode + "/" + split + "_seed_" + str(seed))
            loaded, all_dataset, transformers, fp, kernel_dicts, \
            simboost_drug_target_feats_dict, MF_entities_dict = load_nested_cv_dataset_from_disk(save_dir, K)
        else:
            save_dir = os.path.join(data_dir, featurizer + delim + mode + "/" + split + "_seed_" + str(seed))
            loaded, all_dataset, transformers, fp, kernel_dicts, \
            simboost_drug_target_feats_dict, MF_entities_dict = load_dataset_from_disk(save_dir)
        if loaded:
            return tasks, all_dataset, transformers, fp, kernel_dicts, simboost_drug_target_feats_dict, MF_entities_dict

    dataset_file = os.path.join(data_dir, file_name)
    if featurizer == 'Weave':
        featurizer = jova.feat.WeaveFeaturizer()
    elif featurizer in ['ECFP4', 'KRLS_ECFP4', 'SB_ECFP4', 'MF_ECFP4']:
        featurizer = jova.feat.CircularFingerprint(size=1024, radius=2)
    elif featurizer in ['ECFP8', 'KRLS_ECFP8', 'SB_ECFP8', 'MF_ECFP8']:
        featurizer = jova.feat.CircularFingerprint(size=1024, radius=4)
    elif featurizer == 'GraphConv':
        featurizer = jova.feat.ConvMolFeaturizer()
    elif featurizer == 'GNN':
        featurizer = GNNFeaturizer(radius=gnn_radius)
        gnn_fingerprint = featurizer.fingerprint_dict

    loader = jova.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", protein_field="proteinName",
        source_field='protein_dataset', featurizer=featurizer, prot_seq_dict=prot_seq_dict)
    dataset = loader.featurize(dataset_file, shard_size=8192)

    if mode == 'regression':
        transformers = [
            jova.trans.NormalizationTransformer(
                transform_y=True, dataset=dataset)
        ]
    elif mode == 'classification':
        transformers = [
            jova.trans.BalancingTransformer(transform_w=True, dataset=dataset)
        ]

    print("About to transform data")
    for transformer in transformers:
        dataset = transformer.transform(dataset)

    if feat_label in ['KRLS_ECFP8', 'KRLS_ECFP4']:
        from jova.data.data import compute_similarity_kernel_matrices
        drug_sim_kernel_dict, prot_sim_kernel_dict = compute_similarity_kernel_matrices(dataset)
    elif feat_label in ['SB_ECFP8', 'SB_ECFP4']:
        from jova.data.data import compute_simboost_drug_target_features
        simboost_drug_target_feats_dict = compute_simboost_drug_target_features(dataset, simboost_mf_feats_dict)
    elif feat_label in ['MF_ECFP8', 'MF_ECFP4']:
        from jova.data.data import compute_MF_entities_matrix
        MF_entities_dict = compute_MF_entities_matrix(dataset)

    splitters = {
        'no_split': splits.NoSplit(),
        'index': jova.splits.IndexSplitter(),
        'random': splits.RandomSplitter(split_cold=predict_cold, cold_drug=cold_drug,
                                        cold_target=cold_target, cold_drug_cluster=cold_drug_cluster,
                                        split_warm=split_warm,
                                        prot_seq_dict=prot_seq_dict, threshold=filter_threshold,
                                        oversampled=oversampled,
                                        input_protein=input_protein),
        'scaffold': jova.splits.ScaffoldSplitter(),
        'butina': jova.splits.ButinaSplitter(),
    }
    splitter = splitters[split]
    from jova.data.data import compute_train_val_test_kronrls_mats
    if test:
        kernel_data = None
        train, valid, test = splitter.train_valid_test_split(dataset, seed=seed)

        # process kernel / kronrls data
        if drug_sim_kernel_dict and prot_sim_kernel_dict:
            print('Constructing kernel data')
            kernel_data = compute_train_val_test_kronrls_mats(train, valid, test, drug_sim_kernel_dict,
                                                              prot_sim_kernel_dict)
            kernel_data = _wrap_kernel_data(kernel_data)
        all_dataset = (train, valid, test, kernel_data)
        if reload:
            save_dataset_to_disk(save_dir, train, valid, test, transformers, gnn_fingerprint, drug_sim_kernel_dict,
                                 prot_sim_kernel_dict, simboost_drug_target_feats_dict, kernel_data, MF_entities_dict)
    elif cross_validation:
        fold_datasets = splitter.k_fold_split(dataset, K, seed=seed)
        fold_datasets = list(fold_datasets)

        # process kernel / kronrls data if simulation is in kernel mode
        if drug_sim_kernel_dict and prot_sim_kernel_dict:
            print('Constructing kernel data')
            fold_ds = []
            for fold in fold_datasets:
                kernel_data = compute_train_val_test_kronrls_mats(fold[0], fold[1], fold[2], drug_sim_kernel_dict,
                                                                  prot_sim_kernel_dict)
                kernel_data = _wrap_kernel_data(kernel_data)
                fold_ds.append(list(fold) + [kernel_data])
            fold_datasets = fold_ds
        all_dataset = fold_datasets
        if reload:
            save_nested_cv_dataset_to_disk(save_dir, all_dataset, K, transformers, gnn_fingerprint,
                                           drug_sim_kernel_dict, prot_sim_kernel_dict,
                                           simboost_drug_target_feats_dict,
                                           MF_entities_dict)
    else:
        kernel_data = None
        # not cross validating, and not testing.
        train, valid, test = splitter.train_valid_test_split(dataset, frac_train=0.9, frac_valid=0.1,
                                                             frac_test=0, seed=seed)
        # process kernel / kronrls data
        if drug_sim_kernel_dict and prot_sim_kernel_dict:
            print('Constructing kernel data')
            kernel_data = compute_train_val_test_kronrls_mats(train, valid, test, drug_sim_kernel_dict,
                                                              prot_sim_kernel_dict)
            kernel_data = _wrap_kernel_data(kernel_data)
        all_dataset = (train, valid, test, kernel_data)
        if reload:
            save_dataset_to_disk(save_dir, train, valid, test, transformers, gnn_fingerprint, drug_sim_kernel_dict,
                                 prot_sim_kernel_dict, simboost_drug_target_feats_dict, kernel_data, MF_entities_dict)

    return tasks, all_dataset, transformers, gnn_fingerprint, \
           (drug_sim_kernel_dict, prot_sim_kernel_dict), simboost_drug_target_feats_dict, MF_entities_dict


def _wrap_kernel_data(kernel_data):
    assert isinstance(kernel_data, dict)
    kd, kt, Y, W = kernel_data['train']
    kd_val, kt_val, Y_val, W_val = kernel_data['val']
    kd_test, kt_test, Y_test, W_test = kernel_data['test']
    kdata = {'KD': kd, 'KT': kt, 'Y': Y, 'W': W,
             'KD_val': kd_val, 'KT_val': kt_val, 'Y_val': Y_val, 'W_val': W_val,
             'KD_test': kd_test, 'KT_test': kt_test, 'Y_test': Y_test, 'W_test': W_test}
    return kdata
