# Author: bbrighttaer
# Project: jova
# Date: 11/12/19
# Time: 11:34 PM
# File: attn_helpers.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


class UnimodalAttentionData(object):
    def __init__(self, view_lbl, view_x, layer_num, x, x_prime, wts):
        self.view_lbl = view_lbl
        self.layer_num = layer_num
        self.data = {"raw_x": view_x,
                     "x": x,  # input multihead attention
                     "x_prime": x_prime,  # output of multihead attention
                     "weights": wts}

    def __getitem__(self, idx):
        pass


class MultimodalAttentionData(object):
    def __init__(self):
        self.labels = []
        self._data_xs = {}
        self._registry = []

    def register_data(self, data_xs):
        assert isinstance(data_xs, dict)
        self._data_xs = data_xs

    def joint_attn_forward_hook(self, layer_num, x, x_prime, wts, num_segs):
        x = x.cpu().detach()
        x_prime = x_prime.cpu().detach()
        wts = wts.cpu().detach()

        # split tensors into unimodal data tensors
        xs = torch.split(x, num_segs, 0)
        x_primes = torch.split(x_prime, num_segs, 0)
        wts_lst = torch.split(wts, num_segs, 1)

        # record unimodal attention data
        for lbl, un_x, un_x_prime, un_wts in zip(self.labels, xs, x_primes, wts_lst):
            unimodal_attn_data = UnimodalAttentionData(lbl, self._data_xs[lbl], layer_num, un_x.numpy(),
                                                       un_x_prime.numpy(), un_wts.numpy())
            self._registry.append(unimodal_attn_data)

        # clear data buffer
        self._data_xs.clear()
