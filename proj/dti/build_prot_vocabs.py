# Author: bbrighttaer
# Project: ivpgan
# Date: 10/17/19
# Time: 11:24 PM
# File: build_prot_vocabs.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

from tqdm import tqdm


def split_sequence(sequence, ngram, vocab_dict):
    sequence = '-' + sequence + '='
    words = [vocab_dict[sequence[i:i + ngram]]
             for i in range(len(sequence) - ngram + 1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates a protein dictionary for training embeddings.")

    parser.add_argument('--prot_desc_path',
                        dest='prot_files',
                        action='append',
                        required=True,
                        help='A list containing paths to protein descriptors.')
    parser.add_argument('--ngram',
                        type=int,
                        default=3,
                        help='Length of each segment')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Prints every entry being processed')
    parser.add_argument('--ds_folder',
                        type=str,
                        dest='dataset',
                        help='Dataset folder to store the files')
    args = parser.parse_args()

    word_dict = defaultdict(lambda: len(word_dict))
    proteins = {}

    for file in args.prot_files:
        print("Loading %s" % file)
        df = pd.read_csv(file)
        for row in tqdm(df.itertuples()):
            label = row[1]
            sequence = row[2]
            if args.verbose:
                print("Label={}, Sequence={}".format(label, sequence))
            protein_profile = split_sequence(sequence, args.ngram, word_dict)
            proteins[label] = protein_profile

    print("Saving files...")
    dump_dictionary(proteins, '../../data/{}/proteins.profile'.format(args.dataset))
    dump_dictionary(word_dict, '../../data/{}/proteins.vocab'.format(args.dataset))
    print("Info: vocab size={}, protein profiles saved={}".format(len(word_dict), len(proteins)))
