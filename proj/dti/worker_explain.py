# Author: bbrighttaer
# Project: jova
# Date: 7/17/19
# Time: 6:00 PM
# File: worker_explain.py


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import json
import os

from tqdm import tqdm

aac_one_to_three_dict = {'A': 'Ala',  # Alanine
                         'R': 'Arg',  # Arginine
                         'N': 'Asn',  # Asparagine
                         'D': 'Asp',  # Aspartic Acid
                         'C': 'Cys',  # Cysteine
                         'E': 'Glu',  # Glutamic Acid
                         'Q': 'Gln',  # Glutamine
                         'G': 'Gly',  # Glycine
                         'H': 'His',  # Histidine
                         'I': 'Ile',  # Isoleucine
                         'L': 'Leu',  # Leucine
                         'K': 'Lys',  # Lysine
                         'M': 'Met',  # Methionine
                         'F': 'Phe',  # Phenylalanine
                         'P': 'Pro',  # Proline
                         'S': 'Ser',  # Serine
                         'T': 'Thr',  # Threonine
                         'W': 'Trp',  # Tryptophan
                         'Y': 'Tyr',  # Tyrosine
                         'V': 'Val'  # Valine
                         }


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
    return {k: query_results[k] for k in query_results}


def retrieve_resource(seeds, r_name, r_data, res_names):
    """
    Aggregates train-validation data for analysis.

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
            path = "{}/{}/seed_{}/{}/{}".format(r_name, i, seed, idx, res)
            r = get_resource(path, r_data)
            query_results[res].append(r)
    return {k: query_results[k] for k in query_results}


if __name__ == '__main__':
    folder = "analysis/egfr_cs"
    qualifier = "egfr_1M17"
    files = list(filter(lambda f: qualifier in f and '.json' in f, os.listdir(folder)))
    print('Number of files loaded=', len(files))
    seq_offset = 0
    files.sort()
    results_folder = "results_" + folder + '_explain'
    os.makedirs(results_folder, exist_ok=True)
    for file in tqdm(files):
        print(file)
        with open(os.path.join(folder, file), "r") as f:
            data = json.load(f)

        root_name = file.split(".j")[0]
        metadata = json.loads(root_name)
        seeds = [eval(s) for s in metadata['seeds'].split('-')]
        cv = eval(metadata['cv'])
        if cv:
            data_dict = retrieve_resource_cv(k=5, seeds=seeds, r_name=root_name, r_data=data,
                                             res_names=[
                                                 ("attn_ranking", 0),
                                             ])
        else:
            data_dict = retrieve_resource(seeds=seeds, r_name=root_name, r_data=data,
                                          res_names=[
                                              ("attn_ranking", 0),
                                          ])
        attention_data = data_dict['attn_ranking']
        with open(os.path.join(results_folder, root_name + '.csv'), 'a') as f:
            writer = csv.DictWriter(f, None)
            for i in range(len(attention_data)):
                attn_dt = attention_data[i]
                for n in range(len(attn_dt)):
                    sample_dt = attn_dt[n]
                    y_true = sample_dt['y_true']
                    y_pred = sample_dt['y_pred']
                    rank_data = sample_dt['attn_ranking']
                    rank_data = list(filter(lambda dt: dt is not None, rank_data))
                    csv_row = {'dataset': None, 'y_true': y_true[0], 'y_pred': y_pred[0]}
                    for v, view_dict in enumerate(rank_data):
                        view_lbl = view_dict['view']
                        csv_row[f'view{v + 1}'] = view_lbl
                        predictions = view_dict['predictions']
                        for sample_id in predictions:
                            sample = predictions[sample_id]
                            entity = sample['entity']
                            if isinstance(entity, list):
                                dataset = entity[0]
                                entity = entity[1]
                                csv_row['dataset'] = dataset
                            csv_row[f'entity{v + 1}'] = entity
                            top_segments = sample['rankings']
                            top_residues = None
                            if 'sequence' in sample:
                                sequence = sample['sequence']
                                csv_row['sequence'] = sequence
                                # since it is a protein lets expand the grouped segments (windows)
                                _tps = []
                                _tps_res = []
                                for window in top_segments:
                                    seg = ''
                                    seg_res = ''
                                    for m, trigram in enumerate(window):
                                        if trigram != '<UNK>':
                                            if m == 0:
                                                seg += trigram
                                            else:
                                                seg += trigram[-1]  # since they overlap
                                    offset = sequence.index(seg)
                                    for r, res in enumerate(seg):
                                        seg_res += aac_one_to_three_dict[res] + str(offset + r + 1 + seq_offset) + ' '
                                    _tps.append(seg)
                                    _tps_res.append(seg_res.strip())
                                top_segments = _tps
                                top_residues = _tps_res
                            csv_row[f'top_segments{v + 1}'] = top_segments
                            if top_residues:
                                csv_row['top_residues'] = top_residues
                    if writer.fieldnames is None:
                        writer.fieldnames = list(csv_row.keys())
                        writer.writeheader()
                    writer.writerow(csv_row)
    print('Done!')
