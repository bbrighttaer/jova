# Re-organization of the GNN work in https://academic.oup.com/bioinformatics/article/35/2/309/5050020
# Author: bbrighttaer
# Project: ivpgan
# Date: 10/29/19
# Time: 2:12 AM
# File: gnnfeat.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
from padme.feat import Featurizer
import numpy as np
from collections import defaultdict
from rdkit import Chem


def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [GNNFeaturizer.atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = GNNFeaturizer.bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [GNNFeaturizer.fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(GNNFeaturizer.fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = GNNFeaturizer.edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


class GnnMol(object):

    def __init__(self, mol, fingerprints, adjacency, smiles):
        self.mol = mol
        self.fingerprints = fingerprints
        self.adjacency = adjacency
        self.smiles = smiles

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            assert self.smiles is not None
            return self.smiles == other.smiles
        return False

    def __hash__(self):
        assert self.smiles is not None
        return hash(self.smiles)


class GNNFeaturizer(Featurizer):
    """
    Graph Neural Net.

    Compound featurizer described in https://academic.oup.com/bioinformatics/article/35/2/309/5050020
    This is basically a featurization wrapper of the initial code accompanying the work above.
    """

    name = ["gnn_mol"]
    atom_dict = defaultdict(lambda: len(GNNFeaturizer.atom_dict))
    bond_dict = defaultdict(lambda: len(GNNFeaturizer.bond_dict))
    fingerprint_dict = defaultdict(lambda: len(GNNFeaturizer.fingerprint_dict))
    edge_dict = defaultdict(lambda: len(GNNFeaturizer.edge_dict))

    def __init__(self, radius=2):
        super(GNNFeaturizer, self).__init__()
        self.radius = radius

    def _featurize(self, mol, smiles):
        """
        Featurizes a compound as described in the paper cited above.
        :param mol:
        :param smiles:
        :return:
        """
        mol = Chem.AddHs(mol)  # Consider hydrogens.
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)
        fingerprints = extract_fingerprints(atoms, i_jbond_dict, self.radius)
        adjacency = create_adjacency(mol)
        return GnnMol(mol, fingerprints, adjacency, smiles)

    @classmethod
    def save_featurization_info(cls, save_dir):
        """
        Persists GNN featurization data needed at runtime.

        :param save_dir: folder to save objects.
        :return:
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'fingerprint_dict.pickle'), 'wb') as f:
            pickle.dump(cls.fingerprint_dict, f)
