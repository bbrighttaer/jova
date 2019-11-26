# Author: bbrighttaer
# Project: jova
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
import torch.nn.functional as F
import torch.nn.init as init

from jova.nn.layers import Linear, Conv1d, Conv2d, ConcatLayer, WeaveGather2D
from jova.nn.layers import WeaveGather, WeaveLayer, GraphConvLayer, GraphGather, GraphPool, WeaveBatchNorm, \
    WeaveDropout
from jova.utils.train_helpers import get_activation_func

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


def _construct_embedding_indices(prot_x, protein_profile, device, fill_val):
    """
    Produces the indices for querying the embeddings for training. Since the sequence length may vary from protein to
    protein, they are padded to have the same sizes to enable batch processing. To this end, the embeddings matrix is
    extended to have one row whose values are all zeros. This row's index is used during the indices construction so as
    to ensure that the 'embeddings' that would be queried for the padded regions are actually zeros.

    :param prot_x: list
        The list of proteins in the batch. The structure is: [(dataset_lbl, protein_name)].
        E.g. [(davis, AAA61480.1|CLK1|CLK1), ...]
    :param protein_profile: dict
        Contains the subsequence groups within a protein for retrieving their respective embeddings.
    :param device: torch.device
        cpu / gpu switching
    :param fill_val: int
        The index of the last row in the embeddings matrix which is filled with zeros.
    :return:
        A 3D tensor of embedding indices to be queried for the batch.
    """
    # retrieve protein embedding profiles
    x = [torch.tensor(protein_profile[prot[1]].tolist(), dtype=torch.long).to(device) for prot in prot_x]

    # get the maximum number of sub-sequence groups in the batch
    max_seq = max([len(p) for p in x])

    # pad sequences for batch processing
    x = [F.pad(tensor, (0, 0, 0, max_seq - tensor.shape[0]), value=fill_val) for tensor in x]
    x = torch.stack(x, dim=0).long().to(device)
    return x


class WeaveModel(nn.Module):

    def __init__(self, weave_args, weave_gath_arg, update_pair=False, weave_type='1D', batch_first=False):
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
                weave_gath = WeaveGather2D(*weave_gath_arg, batch_first=batch_first)
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
                    or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU) \
                    or isinstance(module, NonsatActivation):
                input[0] = module(input[0])
            elif isinstance(module, GraphGather):
                input[0] = module(input, batch_size)
            else:
                input[0] = module(input)
        return input[0]


class GraphNeuralNet(nn.Module):
    """
    Wrapper for the GNN work in https://academic.oup.com/bioinformatics/article/35/2/309/5050020
    """

    def __init__(self, num_fingerprints, embedding_dim, num_layers=3, update='mean', activation='relu'):
        super(GraphNeuralNet, self).__init__()
        self.output = update
        self.activation = get_activation_func(activation)
        self.embed_fingerprint = nn.Embedding(num_fingerprints, int(embedding_dim))
        self.W_gnn = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)])

    def update_fingerprints(self, xs, A, M, i):
        """Update the node vectors in a graph
        considering their neighboring node vectors (i.e., sum or mean),
        which are non-linear transformed by neural network."""
        hs = self.activation(self.W_gnn[i](xs))
        if self.output == 'sum':
            return xs + torch.matmul(A, hs)
        else:
            return xs + torch.matmul(A, hs) / (M - 1)

    def forward(self, inputs):
        fingerprints, adjacency_matrices, M, axis = inputs

        fingerprints = self.embed_fingerprint(fingerprints)

        for i in range(len(self.W_gnn)):
            fingerprints = self.update_fingerprints(fingerprints, adjacency_matrices, M, i)

        if self.output == 'sum':
            molecular_vectors = self.sum_axis(fingerprints, axis)
        else:
            molecular_vectors = self.mean_axis(fingerprints, axis)

        return molecular_vectors

    def sum_axis(self, xs, axis):
        y = [torch.sum(x, 0) for x in torch.split(xs, axis)]
        return torch.stack(y)

    def mean_axis(self, xs, axis):
        y = [torch.mean(x, 0) for x in torch.split(xs, axis)]
        return torch.stack(y)


class GraphNeuralNet2D(GraphNeuralNet):

    def forward(self, inputs):
        """

        :param inputs:
        :return: 3D tensor
            Structure: [number of segments, batch size, dimension]
        """
        fingerprints, adjacency_matrices, M, axis = inputs

        fingerprints = self.embed_fingerprint(fingerprints)

        for i in range(len(self.W_gnn)):
            fingerprints = self.update_fingerprints(fingerprints, adjacency_matrices, M, i)

        mols = torch.split(fingerprints, axis)
        max_seg = max([len(m) for m in mols])
        mols = [F.pad(mol, (0, 0, 0, max_seg - len(mol))) for mol in mols]
        mols = torch.stack(mols, 1)
        return mols


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


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features, dprob, activation='relu'):
        super(ResBlock, self).__init__()
        self.activation = get_activation_func(activation)
        self.net1 = nn.Sequential(nn.Linear(in_features, out_features),
                                  nn.BatchNorm1d(out_features),
                                  nn.ReLU(),
                                  nn.Dropout(dprob))
        self.net2 = nn.Sequential(nn.Linear(out_features, out_features),
                                  nn.BatchNorm1d(out_features),
                                  nn.ReLU(),
                                  nn.Dropout(dprob))
        self.bn_out = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.net1(x)
        x = x + self.net2(x)
        x = self.bn_out(x)
        x = self.activation(x)
        return x


class DINA(nn.Module):

    def __init__(self, total_dim, activation=nonsat_activation, heads=1, bias=True):
        super(DINA, self).__init__()
        self.heads = heads
        self.add_bias = bias
        self.batch_norm_2d = nn.BatchNorm2d(1)
        # self.register_parameter('U', None)
        self.U = nn.Parameter(torch.empty((self.heads, total_dim, total_dim)))
        init.xavier_normal_(self.U)
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
        # if self.U is None:
        #     self.U = nn.Parameter(torch.empty((self.heads, c, c)).to(device))
        #     init.xavier_normal_(self.U)

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
            alpha = alpha.view(alpha.shape[0], 1, -1)

            # attention
            reps = []
            offset = 0
            for i, context in enumerate(contexts):
                C = context
                a = alpha[:, :, offset: offset + L[i]]
                offset += L[i]
                wt = torch.softmax(a, dim=2)
                r = wt.bmm(C)
                reps.append(r.view(r.shape[0], -1))
            h_reps.append(torch.stack(reps, dim=1))
        return h_reps

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
        tensor = torch.stack(inputs, dim=1)
        # pooled = []
        # for C in inputs:
        if self.pool == 'max':
            pooled, _ = torch.max(tensor, dim=1)
        else:
            pooled = torch.mean(tensor, dim=1)
            # pooled.append(p.view(C.shape[0], -1))

        # projection
        x = pooled.view(pooled.shape[0], -1)
        x = self.linear(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x


class ProteinRNN(nn.Module):

    def __init__(self, in_dim, hidden_dim, dropout, num_layers=1, bidrectional=False, activation='relu',
                 batch_first=False):
        super(ProteinRNN, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.directions = max(1, int(bidrectional) + 1)
        self.activation = get_activation_func(activation)
        self.batch_first = batch_first
        if num_layers == 1:
            dropout = 0
        self.model = nn.LSTM(input_size=int(in_dim), hidden_size=int(self.hidden_dim), num_layers=int(self.num_layers),
                             batch_first=batch_first, dropout=dropout, bidirectional=bidrectional)

    def forward(self, x):
        # RNN initial states
        # (layer_dim * num_directions, batch_size, hidden_dim)
        batch_sz = x.shape[0] if self.batch_first else x.shape[1]
        h0 = torch.zeros(self.num_layers * self.directions, batch_sz, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * self.directions, batch_sz, self.hidden_dim).to(x.device)

        # forward pass
        output, _ = self.model(x, (h0, c0))
        output = self.activation(output)
        return output


class Prot2Vec(nn.Module):

    def __init__(self, protein_profile, vocab_size, embedding_dim, activation='relu', batch_first=False):
        """

        :param protein_profile:
        :param vocab_size: int
            The number of ngrams formed (excluding padding index). An additional row is added to the end of the
            embedding table constructed as a padding index.
        :param embedding_dim:
        :param activation:
        :param batch_first:
        """
        super(Prot2Vec, self).__init__()
        self._batch_first = batch_first
        self.protein_profile = protein_profile
        self.embedding = nn.Embedding(vocab_size + 1, int(embedding_dim), padding_idx=vocab_size)
        self.activation = get_activation_func(activation)

    def forward(self, input):
        # get the embedding indices for this batch
        fill_val = self.embedding.weight.shape[0] - 1
        x = _construct_embedding_indices(input, self.protein_profile, self.embedding.weight.device, fill_val)

        # get protein embedding
        embedding = self.embedding(x)
        embedding = embedding.reshape(*embedding.shape[:2], -1)
        embedding = self.activation(embedding)
        if not self._batch_first:
            embedding = embedding.permute(1, 0, 2)
        return embedding


class ProteinCNN(nn.Module):
    """
    Implements Protein CNN without Attention
    """

    def __init__(self, dim, window, activation='relu', num_layers=2, pooling_dim=1):
        """

        :param dim: int
            final dimension of the protein representation
        :param activation:
            non-linearity to apply to logits
        :param window:
            max size of grouped amino acids
        :param num_layers: int
            Number of convolution layers
        :param pooling_dim: int
            The dimension to be used in reducing protein segments to form a vector representation of the protein.
        """
        super(ProteinCNN, self).__init__()
        self.activation = get_activation_func(activation)
        self.pooling_dim = pooling_dim
        self.lin_kernels = nn.ModuleList([nn.Linear(dim * window, dim * window) for _ in range(num_layers - 1)])
        self.lin_kernels.append(nn.Linear(dim * window, dim))

    def forward(self, prot_x):
        """
        Protein CNN without attention as described in https://academic.oup.com/bioinformatics/article/35/2/309/5050020
        :param x: tensor,
            3D protein embeddings. [batch_size, number of segments/groups, window * dimension]
            Supplied by the Prot2Vec model.
        :return: 2D tensor [batch_size, dimension]
        """
        # apply convolution
        for kernel in self.lin_kernels:
            prot_x = self.activation(kernel(prot_x))

        # Output: protein vector representation
        prot_x = torch.mean(prot_x, dim=self.pooling_dim, keepdim=True)
        return prot_x


class ProteinCNN2D(ProteinCNN):

    def forward(self, prot_x):
        """
        Protein CNN without attention as described in https://academic.oup.com/bioinformatics/article/35/2/309/5050020
        :param x: 3D protein embeddings.
        :return: 3D tensor
        """
        # apply convolution
        for kernel in self.lin_kernels:
            prot_x = self.activation(kernel(prot_x))
        return prot_x


class ProteinCNNAttention(ProteinCNN):
    """
    Implementation of the Protein CNN (with attention)
    presented in https://academic.oup.com/bioinformatics/article/35/2/309/5050020
    All credits to the work above for the initial implementation which this implementation builds on.
    """

    def __init__(self, dim, activation='relu', window=11, num_layers=3):
        super(ProteinCNNAttention, self).__init__(dim, window, activation, num_layers)
        self.W_attention = nn.Linear(dim, dim)

    def forward(self, prot_x, comp_x):
        """
        Applies the Protein CNN and attention mechanism using the compound representation provided as described
        in the above cited paper.

        :param prot_x: tensor
            protein embeddings.
        :param comp_x: tensor
            A 2D tensor of shape [batch_size, rep_dimension]
        :return: tensor
            A 2D tensor representing the representations of the proteins.
        """

        # apply convolution
        for kernel in self.lin_kernels:
            prot_x = self.activation(kernel(prot_x))

        # Attention
        h_comp = torch.relu(self.W_attention(comp_x)).unsqueeze(1)
        h_prot = torch.relu(self.W_attention(prot_x))
        wts = h_comp.bmm(h_prot.permute(0, 2, 1))
        attn_weights = torch.softmax(wts, dim=2)
        prot_out = attn_weights.permute(0, 2, 1) * h_prot
        prot_out = torch.mean(prot_out, dim=1).reshape(len(prot_x), -1)
        return prot_out


class ProtCnnForward(nn.Module):
    """
    Helper forward propagation module for :class:ProtCNN
    """

    def __init__(self, prot2vec, prot_cnn_model, comp_model):
        """
        Note: The final dimension of the proteins and compounds must be equal due to the PCNN attention calculation.
        :param prot_cnn_model:
            protein model
        :param comp_model:
            compound model
        """
        super(ProtCnnForward, self).__init__()
        self.prot2vec = prot2vec
        self.pcnn = prot_cnn_model
        self.gnet = comp_model

    def forward(self, inputs):
        """
        First get the compound representations and then forward them to the protein CNN.
        This is necessary since the compound features are used in the ProtCNN attention weights calculation.

        :param inputs: list
        :return:
        """
        comp_input, prot_input = inputs
        comp_out = self.gnet(comp_input)
        prot_input = self.prot2vec(prot_input)
        prot_out = self.pcnn(prot_input, comp_out)
        out = torch.cat([comp_out, prot_out], dim=1)
        return out


class TwoWayForward(nn.Module):

    def __init__(self, model1, model2):
        super(TwoWayForward, self).__init__()
        self.models = nn.ModuleList([model1, model2])

    def forward(self, inputs):
        outs = []
        for i, model in enumerate(self.models):
            outs.append(model(inputs[i]))
        return outs


class TwoWayAttention(nn.Module):

    def __init__(self, dim1, dim2, activation=nonsat_activation):
        super(TwoWayAttention, self).__init__()
        self.activation = activation
        self.U = nn.Parameter(torch.empty((dim1, dim2)))
        init.xavier_normal_(self.U)

    def forward(self, inputs):
        x1, x2 = inputs

        batch_sz = x1.shape[0]
        U = self.U.repeat(batch_sz, 1, 1)
        M = x1.bmm(U).bmm(x2.permute(0, 2, 1))
        rows, _ = torch.max(M, dim=2, keepdim=True)
        rows = rows.view(batch_sz, 1, -1)
        rows = torch.softmax(rows, dim=2)
        cols, _ = torch.max(M, dim=1, keepdim=True)
        cols = cols.view(batch_sz, 1, -1)
        cols = torch.softmax(cols, dim=2)
        _x1 = rows.bmm(x1).squeeze()
        _x2 = cols.bmm(x2).squeeze()
        return _x1, _x2


class JointAttention(nn.Module):

    def __init__(self, d_dims, latent_dim=256, num_heads=1, num_layers=1, dprob=0., activation='relu',
                 inner_layer_dim=2048, attn_hook=None):
        super(JointAttention, self).__init__()
        if attn_hook:
            assert callable(attn_hook), "Attention hook must be a function"
        self.attn_hook = attn_hook
        self.latent_dim = latent_dim
        self.lin_prjs = nn.ModuleList([nn.Linear(in_dim, latent_dim) for in_dim in d_dims])
        self.attention_models = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.latent_dim,
                                                                     num_heads=num_heads, dropout=dprob)
                                               for _ in range(num_layers)])
        self.res_blks = nn.ModuleList([SegmentWiseResBlock(d_model=latent_dim, activation=activation,
                                                           inner_layer_dim=inner_layer_dim) for _ in range(num_layers)])
        self.activation = get_activation_func(activation)
        self.dropout = nn.Dropout(dprob)
        self.batch_norm = nn.BatchNorm1d(latent_dim * len(d_dims))
        self.attn_bn = nn.BatchNorm1d(latent_dim)

    @classmethod
    def _pad_tensors(cls, c, contexts):
        padded = []
        for X in contexts:
            mask = F.pad(torch.ones_like(X), (0, c - X.shape[-1])).bool()
            X_padded = torch.zeros(mask.shape).to(X.device)
            X_padded = X_padded.masked_scatter(mask, X)
            padded.append(X_padded)
        return padded

    def forward(self, inputs):
        # Gets number of segments in each view
        num_segs = [x.shape[0] for x in inputs]

        # projection of segments into equal dimension space
        xs = [self.lin_prjs[i](x) for i, x in enumerate(inputs)]

        # join all segments along the 'seq' dimension
        x = torch.cat(xs)
        shape = x.shape

        # self-attention
        for i, (attn_net, res_net) in enumerate(zip(self.attention_models, self.res_blks)):
            # self-attention - sub-module 1
            x_prime, wts = attn_net(x, x, x)

            # call any registered hook
            if self.attn_hook:
                self.attn_hook(i, x, x_prime, wts, num_segs)

            # residual connection (segment-wise)
            x_add = x + x_prime
            x = self.attn_bn(x_add.view(shape[0] * shape[1], shape[2]))
            x = x.view(*shape)

            # FFN res block - sub-module 2
            x = res_net(x)

        # compute view representations
        xs = torch.split(x, num_segs, 0)
        # pooling
        xs = [torch.sum(x, 0) for x in xs]

        # concat view reps
        x = torch.cat(xs, 1)
        x = self.dropout(self.activation(self.batch_norm(x)))
        return x


class SegmentWiseLinear(nn.Module):

    def __init__(self, d_model, activation='relu', inner_layer_dim=2048):
        super(SegmentWiseLinear, self).__init__()
        self.activation = get_activation_func(activation)
        self.linear1 = nn.Linear(d_model, inner_layer_dim)
        self.linear2 = nn.Linear(inner_layer_dim, d_model)

    def forward(self, x):
        """
        x shape: [num_segments, batch_size, d_model]
        :param x: tensor
        :return: output shape: [num_segments, batch_size, d_model]
        """
        num_seg, bsize, d_model = x.shape

        # Get the position/segment-wise tensor
        x = x.view(num_seg * bsize, d_model)

        # apply projections
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = x.view(num_seg, bsize, d_model)
        return x


class SegmentWiseResBlock(nn.Module):

    def __init__(self, d_model, activation='relu', inner_layer_dim=2048):
        super(SegmentWiseResBlock, self).__init__()
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.seg_lin = SegmentWiseLinear(d_model, activation, inner_layer_dim)

    def forward(self, x):
        """
        x shape: [num_segments, batch_size, d_model]
        :param x: tensor
        :return: output shape: [num_segments, batch_size, d_model]
        """
        num_seg, bsize, d_model = x.shape
        x = x + self.seg_lin(x)
        x = self.batch_norm(x.view(num_seg * bsize, d_model)).view(num_seg, bsize, d_model)
        return x


class MatrixFactorization(nn.Module):
    """
    Matrix Factorization for DTI prediction (baseline and SimBoost component)
    """

    def __init__(self, ncomps, nprots, k=10):
        super(MatrixFactorization, self).__init__()
        self._P = nn.Parameter(torch.randn(k, ncomps))
        self._Q = nn.Parameter(torch.randn(k, nprots))

    @property
    def P(self):
        return self._P.cpu().detach()

    @property
    def Q(self):
        return self._Q.cpu().detach()

    def forward(self):
        return self._P.t().mm(self._Q)
