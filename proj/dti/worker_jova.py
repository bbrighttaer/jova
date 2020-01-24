# Author: bbrighttaer
# Project: jova
# Date: 7/17/19
# Time: 6:00 PM
# File: worker.py


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_resources(root, queries):
    """Retrieves a list of resources under a root."""
    q_res = []
    for p in queries:
        res = get_resource(p, root)
        q_res.append(res)
    return q_res


def get_resource(p, root):
    """Retrieves a single resource under a root."""
    els = p.split('/')
    els.reverse()
    res = finder(root, els)
    return res


def finder(res_tree, nodes):
    """
    Uses recursion to locate a leave resource.

    :param res_tree: The (sub)resource containing the desired leave resource.
    :param nodes: A list of nodes leading to the resource.
    :return: Located resource content.
    """
    if len(nodes) == 1:
        return res_tree[nodes[0]]
    else:
        cur_node = nodes.pop()
        try:
            return finder(res_tree[cur_node], nodes)
        except TypeError:
            return finder(res_tree[int(cur_node)], nodes)


def retrieve_resource_cv(k, seeds, r_name, r_data, res_names):
    """
    Aggregates cross-validation data for analysis.

    :param k: number of folds.
    :param seeds: A list of seeds used for the simulation.
    :param r_name: The name of the root resource.
    :param r_data: The json data.
    :param res_names: A list resource(s) under each fold to be retrieved.
                      Each record is a tuple of (leave resource path, index of resource path under the given CV fold)
    :return: A dict of the aggregated resources across seeds and folds.
    """
    query_results = dict()
    for res, idx in res_names:
        query_results[res] = []
        for i, seed in enumerate(seeds):
            for j in range(k):
                path = "{}/{}/seed_{}cv/{}/fold-{}/{}/{}".format(r_name, i, seed, j, j, idx, res)
                r = get_resource(path, r_data)
                query_results[res].append(r)
    return {k: np.array(query_results[k]) for k in query_results}


def map_name(lbl):
    return {'weave': 'Weave',
            'gconv': 'GraphConv'}.get(lbl, lbl.upper())


if __name__ == '__main__':
    aggregation_dict = defaultdict(lambda: pd.DataFrame({'model': [], 'split': [], 'metric': [], 'value': [],
                                                         'stdev': [], 'mode': [], 'date': [], 'seeds': []}))
    chart_type = "png"
    folder = "analysis/kiba/eval/baselines"
    qualifier = 'kiba'
    files = [f for f in os.listdir(folder) if qualifier in f and ".json" in f]
    print('Number of files loaded=', len(files))
    files.sort()
    results_folder = None
    for file in files:
        sns.set_style("darkgrid")

        print(file)
        with open(os.path.join(folder, file), "r") as f:
            data = json.load(f)

        root_name = file.split(".j")[0]
        metadata = json.loads(root_name)
        seeds = [eval(s) for s in metadata['seeds'].split('-')]
        data_dict = retrieve_resource_cv(k=5, seeds=seeds, r_name=root_name, r_data=data,
                                         res_names=[
                                             ("validation_metrics/nanmean-rms_score", 0),
                                             ("validation_metrics/nanmean-concordance_index", 0),
                                             ("validation_metrics/nanmean-pearson_r2_score", 0),
                                             ("validation_score", 1),
                                             ("model_predictions/y", 2),
                                             ("model_predictions/y_pred", 2)
                                         ])
        for k in data_dict:
            print('\t', k, data_dict[k].shape)
        print()

        # aggregate data
        dataframe = aggregation_dict[metadata['dataset']]
        mode = metadata['mode']
        split = metadata['split'].replace('_', ' ')
        date = metadata['date']
        model = {'cpi': lambda: 'CPI',
                 'kronrls': lambda: 'KronRLS',
                 'simboost': lambda: 'SimBoost',
                 'singleview': lambda: f'{map_name(metadata["cview"])}-{map_name(metadata["pview"])}',
                 'IntView': lambda: 'IntView',
                 'jova': lambda: f'{"-".join([map_name(lbl) for lbl in metadata["cviews"].split("-")])}'
                                 f'-{"-".join([map_name(lbl) for lbl in metadata["pviews"].split("-")])}',
                 'ivpgan': lambda: 'IVPGAN',
                 '2way-dti': lambda: '2Way-DTI'}.get(metadata['model_family'], lambda: 'label-not-found')()
        if results_folder is None:
            results_folder = "jova_results_" + folder + '_' + chart_type
            os.makedirs(results_folder, exist_ok=True)

        # calculate avg rms
        rms_mean = np.nanmean(data_dict["validation_metrics/nanmean-rms_score"])
        rms_std = np.nanstd(data_dict["validation_metrics/nanmean-rms_score"])
        rms_mean_std = "RMSE: mean={:.4f}, std={:.3f}".format(rms_mean, rms_std)
        dataframe = dataframe.append({'model': model,
                                      'split': split,
                                      'metric': 'RMSE',
                                      'value': rms_mean,
                                      'stdev': rms_std,
                                      'mode': mode,
                                      'date': date,
                                      'seeds': metadata['seeds']}, ignore_index=True)
        aggregation_dict[metadata['dataset']] = dataframe
        print('\t', rms_mean_std)

        # calculate avg ci
        ci_mean = np.nanmean(data_dict["validation_metrics/nanmean-concordance_index"])
        ci_std = np.nanstd(data_dict["validation_metrics/nanmean-concordance_index"])
        ci_mean_std = "CI: mean={:.4f}, std={:.3f}".format(ci_mean, ci_std)
        dataframe = dataframe.append({'model': model,
                                      'split': split,
                                      'metric': 'CI',
                                      'value': ci_mean,
                                      'stdev': ci_std,
                                      'mode': mode,
                                      'date': date,
                                      'seeds': metadata['seeds']}, ignore_index=True)
        aggregation_dict[metadata['dataset']] = dataframe
        print('\t', ci_mean_std)

        # calculate avg r2
        r2_mean = np.nanmean(data_dict["validation_metrics/nanmean-pearson_r2_score"])
        r2_std = np.nanstd(data_dict["validation_metrics/nanmean-pearson_r2_score"])
        r2_mean_std = "R2: mean={:.4f}, std={:.3f}".format(r2_mean, r2_std)
        dataframe = dataframe.append({'model': model,
                                      'split': split,
                                      'metric': 'R2',
                                      'value': r2_mean,
                                      'stdev': r2_std,
                                      'mode': mode,
                                      'date': date,
                                      'seeds': metadata['seeds']}, ignore_index=True)
        aggregation_dict[metadata['dataset']] = dataframe
        print('\t', r2_mean_std)

        with open(os.path.join(results_folder, root_name + '.txt'), "w") as txt_file:
            txt_file.writelines([rms_mean_std + '\n', ci_mean_std + '\n', r2_mean_std])

        # plot and save prediction and joint plots for this root to file (w.r.t data set).
        fig, ax = plt.subplots()
        y_true = data_dict["model_predictions/y"][0]  # we select one of the predictions
        y_pred = data_dict["model_predictions/y_pred"][0]
        # for i, (y1, y2) in enumerate(zip(y_true, y_pred)):
        data = pd.DataFrame({"true value": y_true,
                             "predicted value": y_pred})
        sns.relplot(x="true value", y="true value", ax=ax, data=data, kind='line', color='r') \
            # .set_axis_labels(
        # "predicted value",
        # "true value")
        f1 = sns.relplot(x="predicted value", y="true value", ax=ax, data=data)
        # f1.set_axis_labels("predicted value", "true value")
        # f1.set(xlabel="predicted value", ylabel="true value")
        fig.savefig("./{}/{}_true-vs-pred.{}".format(results_folder, root_name, chart_type))

        sns.set_style("white")
        f2 = sns.jointplot(x="predicted value", y="true value", data=data, kind='kde')  # , stat_func=pearsonr)
        # f2.annotate(pearsonr)
        # f2.set_axis_labels("predicted value", "true value")
        f2.savefig("./{}/{}_joint.{}".format(results_folder, root_name, chart_type))
        plt.close('all')

        # plot neighborhood histogram
        # plt.title(title)
        array_h1, array_v1 = np.meshgrid(y_true, y_true)
        array_nvals1 = np.sort(np.abs(array_v1 - array_h1), axis=1)[:, 1:]
        plt.hist(array_nvals1.reshape(-1, ), bins=20, label='Ground truth', alpha=0.5)

        array_h2, array_v2 = np.meshgrid(y_pred, y_pred)
        array_nvals2 = np.sort(np.abs(array_v2 - array_h2), axis=1)[:, 1:]
        plt.hist(array_nvals2.reshape(-1, ), bins=20, label='Predicted', alpha=0.5)

        plt.legend(loc='best')
        plt.savefig(os.path.join(results_folder, root_name + 'neighborhood.' + chart_type))
        plt.close('all')

        # Distribution plot of the same neighborhood distances
        sns.set(color_codes=True)
        sns.set_style('ticks')
        sns.kdeplot(array_nvals1.reshape(-1, ), shade=True, label='Ground truth')
        sns.kdeplot(array_nvals2.reshape(-1, ), shade=True, label='Predicted')
        sns.despine(offset=5, trim=True)
        plt.savefig(os.path.join(results_folder, root_name + 'neighborhood_kde.' + chart_type))

        plt.close('all')

        print('-' * 100)
    for ds in aggregation_dict:
        aggregation_dict[ds].to_csv(os.path.join(results_folder, ds + '.csv'), index=False)
