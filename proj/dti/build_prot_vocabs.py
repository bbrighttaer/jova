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


def create_words(sequence, ngram=3, offsets=(0,)):
    all_words = []
    for offset in offsets:
        words = [sequence[i:i + ngram]
                 for i in range(offset, len(sequence) - offset - ngram + 1)]
        all_words += words
    return all_words


def split_sequence_overlapping(sequence, ngram=3):
    words = [sequence[i:i + ngram]
             for i in range(len(sequence) - ngram + 1)]
    return words


def create_protein_profile(vocab, words):
    profile = [vocab[w] for w in words]
    return np.array(profile)


def dump_binary(obj, filename, clazz):
    with open(filename, 'wb') as f:
        pickle.dump(clazz(obj), f)


def load_binary(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


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
    parser.add_argument('--vocab',
                        type=str,
                        required=True,
                        help='The file containing protein words (keys) and their index (values) '
                             'in the ProtVec embeddings')
    args = parser.parse_args()

    word_dict = load_binary(args.vocab)
    proteins = {}

    for file in args.prot_files:
        print("Loading %s" % file)
        df = pd.read_csv(file)
        for row in tqdm(df.itertuples()):
            label = row[1]
            sequence = row[2]
            if args.verbose:
                print("Label={}, Sequence={}".format(label, sequence))
            words = split_sequence_overlapping(sequence, args.ngram)
            protein_profile = create_protein_profile(word_dict, words)
            proteins[label] = protein_profile
    print("Saving files...")
    dump_binary(proteins, '../../data/protein/proteins.profile', dict)
    # dump_binary(word_dict, '../../data/protein/proteins.vocab', dict)
    print("Info: vocab size={}, protein profiles saved={}".format(len(word_dict), len(proteins)))
