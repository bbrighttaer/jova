# Author: bbrighttaer
# Project: jova
# Date: 11/12/19
# Time: 11:34 PM
# File: attn_helpers.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import rdkit.Chem as Chem
import torch
from torch.nn.functional import pad


class UnimodalAttentionData(object):
    def __init__(self, view_lbl, view_x, prot_seqs, prot_profile, prot_vocab, layer_num, x, x_prime, wts):
        self.view_lbl = view_lbl
        self._unk = '<UNK>'
        self.prot_profile = prot_profile
        self.prot_seqs = prot_seqs
        self.prot_vocab = prot_vocab
        self.prot_vocab_flipped = {k: v for k, v in zip(prot_vocab.values(), prot_vocab.keys())}
        self.prot_vocab_flipped[len(prot_vocab)] = self._unk  # for padded regions
        self.layer_num = layer_num
        self.raw_x = view_x
        self.x = x  # input attention, (num_segs, batch_size, dimension)
        self.x_prime = x_prime  # output of attention (num_segs, batch_size, dimension)
        self.weights = wts  # (batch_size, num_segs, num_segs)

    def _get_x_prime_top_k(self, k):
        x_prime_norm = torch.norm(self.x_prime, dim=2)
        ranked = torch.sort(x_prime_norm, dim=0, descending=True)[1]
        return ranked[0:k]

    def _compound_segments_ranking(self, k):
        indices = self._get_x_prime_top_k(k).t().numpy()
        compounds = []
        n_atoms_lst = []
        smiles_lst = []
        for pair in self.raw_x:
            mol, _ = pair
            smiles_lst.append(mol.smiles)
            mol = Chem.MolFromSmiles(mol.smiles)
            n_atoms_lst.append(mol.GetNumAtoms())
            compounds.append([(atom.GetSymbol(), atom.GetIdx()) for atom in mol.GetAtoms()])
        max_seg = max(n_atoms_lst)
        compounds = [atoms + [self._unk] * (max_seg - len(atoms)) for atoms in compounds]
        compounds = np.array(compounds, dtype=np.object)
        # assert (compounds.transpose().shape == self.x.shape[:2])
        assert len(indices) == len(compounds) == len(smiles_lst)

        res_dict = {'view': self.view_lbl, 'predictions': {}}
        for i, (idx, comp, smiles) in enumerate(zip(indices, compounds, smiles_lst)):
            samp_dict = {'entity': smiles, 'rankings': comp[idx].tolist()}
            res_dict['predictions']['sample_%d' % i] = samp_dict
        return res_dict

    def _protein_segments_ranking(self, k):
        indices = self._get_x_prime_top_k(k).t().long()  # structure is [batch, num_seg]

        # retrieve protein profiles
        x = [torch.tensor(self.prot_profile[prot[1]].tolist(), dtype=torch.long) for prot in self.raw_x]

        # get the maximum number of sub-sequence groups in the batch
        max_seq = max([len(p) for p in x])

        # pad sequences for batch processing
        x = [pad(tensor, (0, 0, 0, max_seq - tensor.shape[0]), value=len(self.prot_vocab)) for tensor in x]
        x = torch.stack(x, dim=0).long()  # structure as [batch, num_seg, dim]
        # assert (x.permute(1, 0, 2).shape[:2] == self.x.shape[:2])

        res_dict = {'view': self.view_lbl, 'predictions': {}}
        assert len(indices) == len(x) == len(self.raw_x)
        for i, (idx, sample, prot) in enumerate(zip(indices, x, self.raw_x)):
            sample_seqs = []
            for sel_seg in sample[idx]:
                sample_seqs.append([self.prot_vocab_flipped[int(id)] for id in sel_seg])
            samp_dict = {'entity': prot, 'sequence': self.prot_seqs[prot][1], 'rankings': sample_seqs}
            res_dict['predictions']['sample_%d' % i] = samp_dict
        return res_dict

    def rank_segments(self, k=10):
        """
        Determines the top-k most influential segments in the prediction as determined by the joint attention module.
        :return:
        """
        # no ranking cannot be done for views with one segment (e.g. ecfp)
        if self.x.dim() == 3:
            if self.view_lbl in ['gconv', 'gnn', 'weave']:
                return self._compound_segments_ranking(k)
            elif self.view_lbl in ['p2v', 'rnn', 'pcnn']:
                return self._protein_segments_ranking(k)
        return None


class MultimodalAttentionData(object):
    def __init__(self):
        self.labels = []
        self._data_xs = {}
        self._registry = []
        self._protein_profile = None
        self._protein_vocab = None
        self._prot_sequences = None

    @property
    def protein_profile(self):
        return self._protein_profile

    @protein_profile.setter
    def protein_profile(self, profile):
        assert isinstance(profile, dict)
        self._protein_profile = profile

    @property
    def protein_vocabulary(self):
        return self._protein_vocab

    @protein_vocabulary.setter
    def protein_vocabulary(self, vocab):
        assert isinstance(vocab, dict)
        self._protein_vocab = vocab

    @property
    def protein_sequences(self):
        return self._prot_sequences

    @protein_sequences.setter
    def protein_sequences(self, p_dict):
        assert isinstance(p_dict, dict)
        self._prot_sequences = p_dict

    def register_data(self, data_xs):
        assert isinstance(data_xs, dict)
        self._data_xs = data_xs

    def joint_attn_forward_hook(self, layer_num, x, x_prime, wts, num_segs):
        self._registry.clear()
        x = x.cpu().detach()
        x_prime = x_prime.cpu().detach()
        wts = wts.cpu().detach()

        # split tensors into unimodal data tensors
        xs = torch.split(x, num_segs, 0)
        x_primes = torch.split(x_prime, num_segs, 0)
        wts_lst = torch.split(wts, num_segs, 1)

        # record unimodal attention data
        for lbl, un_x, un_x_prime, un_wts in zip(self.labels, xs, x_primes, wts_lst):
            unimodal_attn_data = UnimodalAttentionData(lbl, self._data_xs[lbl], self.protein_sequences,
                                                       self.protein_profile, self.protein_vocabulary,
                                                       layer_num, un_x, un_x_prime, un_wts)
            self._registry.append(unimodal_attn_data)

        # clear data buffer
        self._data_xs.clear()

    def get_rankings(self, k=10):
        return [unimodal.rank_segments(k) for unimodal in self._registry]
