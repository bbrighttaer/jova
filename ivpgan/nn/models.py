# Author: bbrighttaer
# Project: ivpgan
# Date: 5/29/19
# Time: 4:19 PM
# File: models.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from ivpgan.nn.layers import Linear, Conv1d, Conv2d, ConcatLayer, WeaveGather2D
from ivpgan.nn.layers import WeaveGather, WeaveLayer, GraphConvLayer, GraphGather, GraphPool, WeaveBatchNorm, \
    WeaveDropout
from ivpgan.utils.train_helpers import get_activation_func

relu_batch_norm = False


def get_weights_init(a=5):
    def init_func(m):
        """
        Initializes the trainable parameters.

        :param m: The submodule object
        """
        if isinstance(m, Linear) or isinstance(m, Conv1d) or isinstance(m, Conv2d):
            # if m.activation_name:
            #     func_name = m.activation_name.split('(')[0].lower()
            #     if func_name in ['sigmoid', 'tanh']:
            #         init.xavier_uniform_(m.weight)
            #     else:
            #         init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            # else:
            init.kaiming_uniform_(m.weight, a=math.sqrt(a))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
                # init.constant_(m.bias, 0)

    return init_func


def create_conv_layers(conv_args):
    layers = []
    for conv_arg in conv_args:
        if conv_arg.conv_type in ["1D", '1d']:
            conv = Conv1d(*conv_arg.args)
            layers.append(conv)

            # Batch normalization
            if conv_arg.use_batch_norm:
                bn = nn.BatchNorm1d(conv_arg[1])
                layers.append(bn)

            # Activation
            if conv_arg.activation_function:
                conv.activation_name = str(conv_arg.activation_function)
                if relu_batch_norm:
                    # if batch norm + relu, do batch norm after applying relu.
                    if conv_arg.use_batch_norm and 'relu' in conv.activation_name.lower():
                        # bn = layers.pop()
                        layers.append(conv_arg.activation_function)
                        # layers.append(bn)
                    else:
                        layers.append(conv_arg.activation_function)
                else:
                    layers.append(conv_arg.activation_function)

            # Dropout
            if conv_arg.dropout > 0:
                dr = nn.Dropout(conv_arg.dropout)
                layers.append(dr)

            # pooling
            if conv_arg.pooling:
                pool = {'max_pool': lambda kwargs: nn.MaxPool1d(**kwargs),
                        'avg_pool': lambda kwargs: nn.AvgPool1d(**kwargs)
                        }.get(conv_arg.pooling.ptype.lower(), lambda x: None)(conv_arg.pooling.kwargs)
                if pool:
                    layers.append(pool)

        elif conv_arg.conv_type in ["2D", '2d']:
            conv = Conv2d(*conv_arg.args)
            layers.append(conv)

            # Batch normalization
            if conv_arg.use_batch_norm:
                bn = nn.BatchNorm2d(conv_arg[1])
                layers.append(bn)

            # Activation
            if conv_arg.activation_function:
                conv.activation_name = str(conv_arg.activation_function)
                if relu_batch_norm:
                    # if batch norm + relu, do batch norm after applying relu.
                    if conv_arg.use_batch_norm and 'relu' in conv.activation_name.lower():
                        # bn = layers.pop()
                        layers.append(conv_arg.activation_function)
                        # layers.append(bn)
                    else:
                        layers.append(conv_arg.activation_function)
                else:
                    layers.append(conv_arg.activation_function)

            # Dropout
            if conv_arg.dropout > 0:
                dr = nn.Dropout2d(conv_arg.dropout)
                layers.append(dr)

            # pooling
            if conv_arg.pooling:
                pool = {'max_pool': lambda kwargs: nn.MaxPool2d(**kwargs),
                        'avg_pool': lambda kwargs: nn.AvgPool2d(**kwargs)
                        }.get(conv_arg.pooling.ptype.lower(), None)(conv_arg.pooling.kwargs)
                if pool:
                    layers.append(pool)
    return layers


def create_fcn_layers(fcn_args):
    layers = []
    for fcn_arg in fcn_args:
        assert fcn_arg.args[
                   1] > 0, "Output layer nodes number must be specified for hidden layers."
        linear = Linear(*fcn_arg.args)
        layers.append(linear)

        # Batch normalization
        if fcn_arg.use_batch_norm:
            bn = nn.BatchNorm1d(fcn_arg[1])
            layers.append(bn)

        # Activation
        if fcn_arg.activation_function:
            linear.activation_name = str(fcn_arg.activation_function)
            if relu_batch_norm:
                # if batch norm + relu, do batch norm after applying relu.
                if fcn_arg.use_batch_norm and 'relu' in linear.activation_name.lower():
                    # bn = layers.pop()
                    layers.append(fcn_arg.activation_function)
                    # layers.append(bn)
                else:
                    layers.append(fcn_arg.activation_function)
            else:
                layers.append(fcn_arg.activation_function)

        # Dropout
        if fcn_arg.dropout > 0:
            dr = nn.Dropout(fcn_arg.dropout)
            layers.append(dr)
    return layers


def create_weave_layers(weave_args, update_pair):
    layers = []
    for weave_arg in weave_args:
        weave = WeaveLayer(*weave_arg.args)
        layers.append(weave)

        # Batch normalization
        if weave_arg.use_batch_norm:
            bn = WeaveBatchNorm(atom_dim=weave_arg[2], pair_dim=weave_arg[3], update_pair=update_pair)
            layers.append(bn)

        # Dropout
        if weave_arg.dropout > 0:
            dr = WeaveDropout(weave_arg.dropout, update_pair=update_pair)
            layers.append(dr)
    return layers


def create_graph_conv_layers(gconv_args):
    layers = []
    for gc_arg in gconv_args:
        gconv = GraphConvLayer(*gc_arg.args)
        layers.append(gconv)

        # Batch normalization
        if gc_arg.use_batch_norm:
            bn = nn.BatchNorm1d(gc_arg[1])
            layers.append(bn)

        # Dropout
        if gc_arg.dropout > -1:
            dr = nn.Dropout(gc_arg.dropout)
            layers.append(dr)

        # Pooling
        if gc_arg.graph_pool:
            p = GraphPool(gc_arg[2], gc_arg[3])
            layers.append(p)

        # Dense layer & normalization & dropout
        layers.append(nn.Linear(gc_arg[1], gc_arg.dense_layer_size))
        layers.append(nn.BatchNorm1d(gc_arg.dense_layer_size))
        if gc_arg.dropout > -1:
            layers.append(nn.Dropout(gc_arg.dropout))

        # Gather
        layers.append(GraphGather())
    return layers


class WeaveModel(nn.Module):

    def __init__(self, weave_args, weave_gath_arg, update_pair=False, weave_type='1D'):
        """
        Creates a weave model

        :param weave_args: A list of weave arguments.
        :param weave_gath_arg: A weave gather argument.
        :param update_pair: Whether to return the pair-wise embeddings.
        """
        super(WeaveModel, self).__init__()
        self.update_pair = update_pair
        layers = create_weave_layers(weave_args, update_pair)
        if weave_gath_arg:
            if weave_type.lower() == '1d':
                weave_gath = WeaveGather(*weave_gath_arg.args)
            else:
                weave_gath = WeaveGather2D(*weave_gath_arg)
            layers.append(weave_gath)
        self.weave_seq = WeaveSequential(*layers)
        # in_dim = weave_args[-1][2]
        # out_dim = weave_gath_arg[1]
        # self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, input, need_pair_feat=False):
        """

        :param need_pair_feat: Whether to return features of atom-atom pairs.
        :param input: The input structure is: [atom_features, pair_features, pair_split, atom_split, atom_to_pair]
        :return: Features of molecules.
        """
        output = self.weave_seq(input)
        if need_pair_feat:
            return output
        return output[0]


class WeaveSequential(nn.Sequential):

    def __init__(self, *args):
        super(WeaveSequential, self).__init__(*args)

    def forward(self, input):
        """
        Forward propagation through all attached layers.

        :param input: The input structure is: [atom_features, pair_features, pair_split, atom_split, atom_to_pair]
        :return: A tuple of atom features and pair features of the last weave layer. (A, P)
        """
        input = list(input)
        A = P = None
        for module in self._modules.values():
            if A is not None:
                input[0] = A
            if P is not None:
                input[1] = P
            if isinstance(module, WeaveBatchNorm) or isinstance(module, WeaveDropout):
                A, P = module(A, P)
            elif isinstance(module, WeaveGather):
                return module([A, P, *input[4:]])  # returns the molecule features
            else:
                A, P = module(input)
        return A, P


class GraphConvModel(nn.Module):

    def __init__(self, conv_args):
        """
        Creates a graph convolution model.

        :param conv_args: a list of convolution layer arguments.
        """
        super(GraphConvModel, self).__init__()
        self.model = GraphConvSequential(*create_graph_conv_layers(conv_args))

    def forward(self, *input):
        """

        :param input: The structure: [standard graph conv list, batch size]
        :return: molecule(s) features.
        """
        output = self.model(*input)
        return output


class GraphConvSequential(nn.Sequential):

    def __init__(self, *args):
        super(GraphConvSequential, self).__init__(*args)

    def forward(self, input):
        batch_size = input[1]
        input = input[0]
        input = list(input)
        for module in self._modules.values():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.Dropout) \
                    or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU):
                input[0] = module(input[0])
            elif isinstance(module, GraphGather):
                input[0] = module(input, batch_size)
            else:
                input[0] = module(input)
        return input[0]


class PairSequential(nn.Module):
    """Handy approach to manage protein and compound models"""

    def __init__(self, mod1: tuple, mod2: tuple, civ_dim=1):
        super(PairSequential, self).__init__()
        self.comp_tup = nn.ModuleList(mod1)
        self.prot_tup = nn.ModuleList(mod2)
        self.civ = ConcatLayer(dim=civ_dim)

    def forward(self, inputs):
        comp_input, prot_input = inputs

        # compound procession
        comp_out = comp_input
        for module in self.comp_tup:
            comp_out = module(comp_out)

        # protein processing
        prot_out = prot_input
        for module in self.prot_tup:
            prot_out = module(prot_out)

        # form a single representation
        output = self.civ((comp_out, prot_out))
        return output


class NonsatActivation(nn.Module):
    def __init__(self, ep=1e-4, max_iter=100):
        super(NonsatActivation, self).__init__()
        self.ep = ep
        self.max_iter = max_iter

    def forward(self, x):
        return nonsat_activation(x, self.ep, self.max_iter)


def nonsat_activation(x, ep=1e-4, max_iter=100):
    """
    Implementation of the Non-saturating nonlinearity described in http://proceedings.mlr.press/v28/andrew13.html

    :param x: float, tensor
        Function input
    :param ep:float, optional
        Stop condition reference point.
    :param max_iter: int, optional,
        Helps to avoid infinite iterations.
    :return:
    """
    y = x.detach().clone()
    i = 0
    while True:
        y_ = (2. * y ** 3. / 3. + x) / (y ** 2. + 1.)
        if torch.mean(torch.abs(y_ - y)) <= ep or i > max_iter:
            return y_
        else:
            i += 1
            y = y_.detach()


class DINA(nn.Module):

    def __init__(self, activation=torch.relu, heads=1, bias=True):
        super(DINA, self).__init__()
        self.heads = heads
        self.add_bias = bias
        self.batch_norm_2d = nn.BatchNorm2d(1)
        self.register_parameter('U', None)
        self.activation = activation
        self._mask_mul = None
        self._mask_add = None

    @classmethod
    def _pad_tensors(cls, c, contexts):
        padded = []
        for X in contexts:
            mask = F.pad(torch.ones_like(X), (0, c - X.shape[-1])).bool()
            X_padded = torch.zeros(mask.shape).to(X.device)
            X_padded = X_padded.masked_scatter(mask, X)
            padded.append(X_padded)
        return padded

    def forward(self, contexts):
        """
        Applies a Direction-Invariant N-way Attention to the given contexts.

        :param contexts: tuple
            The N contexts for computing the attention vectors.
            Each element's shape must be (batch_size, l, d)
        :return: tuple
            attention vectors.
        """
        L = [c.shape[1] for c in contexts]
        c = max([m.shape[-1] for m in contexts])
        q = sum(L)
        scaling = c ** -0.5
        device = contexts[0].device

        # pad where necessary
        contexts = self._pad_tensors(c, contexts)

        # initialize trainable parameter
        if self.U is None:
            self.U = nn.Parameter(torch.empty((self.heads, c, c)).to(device))
            init.xavier_normal_(self.U)

        # padding
        h_reps = []
        M = torch.cat(contexts, dim=1).to(device)
        # todo(bbrighttaer): parallelize
        for h in range(self.heads):
            U = torch.stack([self.U[h, :, :].squeeze()] * M.shape[0], dim=0)
            K = M.bmm(U).bmm(M.permute(0, 2, 1))
            # normalize each sample
            # K = self.batch_norm_2d(K.unsqueeze(dim=1)).squeeze()
            K = K * scaling
            K = self.activation(K)

            # masking
            if self._mask_mul is None or q != self._mask_mul.shape[1]:
                self._mask_mul = self._diag_sub_mat_mask(dim=q, sizes=L[1:], dvc=M.device, diag_fill_val=0)
                self._mask_add = self._diag_sub_mat_mask(dim=q, sizes=L[1:], dvc=M.device,
                                                         diag_fill_val=torch.min(K).item(),
                                                         off_diag_fill_func=torch.zeros)
            K_hat = K * self._mask_mul
            K_hat = K_hat + self._mask_add

            # max pooling
            cols, _ = torch.max(K_hat, dim=1)
            rows, _ = torch.max(K_hat, dim=2)
            alpha = rows + cols

            # attention
            reps = []
            offset = 0
            for i, context in enumerate(contexts):
                C = context
                a = alpha[:, offset: offset + L[i]]
                offset += L[i]
                wt = torch.softmax(a, dim=1)
                wt = wt.unsqueeze(dim=1)
                r = wt.bmm(C).squeeze()
                reps.append(r)
            h_reps.append(reps)
        c_hat = []
        for i in range(len(contexts)):
            c_hat.append(torch.stack([c_lst[i] for c_lst in h_reps], dim=0).permute(1, 0, 2))
        return c_hat

    @classmethod
    def _diag_sub_mat_mask(cls, dim, sizes, dvc, off_diag_fill_func=torch.ones, diag_fill_val=-999):
        D = off_diag_fill_func(dim, dim).to(dvc)
        prev = 0
        for s in sizes:
            D[prev:prev + s, prev:prev + s] = diag_fill_val
            prev += s
        return D


class NwayForward(nn.Module):

    def __init__(self, models):
        super(NwayForward, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, inputs):
        outs = []
        for i, model in enumerate(self.models):
            outs.append(model(inputs[i]))
        return outs


class Projector(nn.Module):
    """
    Takes a list of 3D inputs and returns a list of 2D inputs.
    """

    def __init__(self, in_features, out_features, bias=True, pool='avg', batch_norm=True, activation='relu'):
        super(Projector, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_features)
        self.activation = get_activation_func(activation)
        self.pool = pool

    def forward(self, inputs):
        """
        Performs pooling on each context to construct the representation and projects all representations of the
        different modalities to a unified space.

        :param inputs: list
            A list of unimodal representations: [(batch_size, segments, latent_dimension),...]
        :return: tensor
            The representations resulting from the projection. shape: [batch_size, len(inputs)*latent_dimension]
        """
        # pooling
        pooled = []
        for C in inputs:
            if self.pool == 'max':
                p, _ = torch.max(C, dim=1)
            else:
                p = torch.mean(C, dim=1)
            pooled.append(p)

        # projection
        X = torch.cat(pooled, dim=1)
        X = self.linear(X)
        if self.batch_norm:
            X = self.bn(X)
        X = self.activation(X)
        return X


class ProteinFeatLearning(nn.Module):

    def __init__(self, protein_profile, vocab_size, embedding_dim, hidden_dim, dropout, num_layers=1,
                 bidrectional=False, activation='nonsat'):
        super(ProteinFeatLearning, self).__init__()
        self.protein_profile = protein_profile
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.directions = max(1, int(bidrectional) + 1)
        self.activation = get_activation_func(activation)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if num_layers == 1:
            dropout = 0
        self.model = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                             dropout=dropout, bidirectional=bidrectional)

    def forward(self, input):
        # retrieve protein embedding profiles
        x = [self.protein_profile[prot[1]].tolist() for prot in input]

        # pad sequences
        max_seq = max([len(p) for p in x])
        x = [vec + [0] * (max_seq - len(vec)) for vec in x]
        x = torch.tensor(x, dtype=torch.long).to(self.embedding.weight.device)

        # get protein embeddings
        embeds = self.embedding(x)

        # RNN initial states
        # (layer_dim * num_directions, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers * self.directions, x.size(0), self.hidden_dim).to(embeds.device)
        c0 = torch.zeros(self.num_layers * self.directions, x.size(0), self.hidden_dim).to(embeds.device)

        # forward pass
        output, _ = self.model(embeds, (h0, c0))
        output = self.activation(output)
        return output
