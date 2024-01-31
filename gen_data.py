import deepchem as dc
import os

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from rdkit import Chem

from deepchem.molnet import load_pdbbind
from deepchem.feat import AtomicConvFeaturizer

f1_num_atoms = 100  # maximum number of atoms to consider in the ligand
f2_num_atoms = 1500  # maximum number of atoms to consider in the protein
max_num_neighbors = 12  # maximum number of spatial neighbors for an atom

acf = AtomicConvFeaturizer(frag1_num_atoms=f1_num_atoms,
                      frag2_num_atoms=f2_num_atoms,
                      complex_num_atoms=f1_num_atoms+f2_num_atoms,
                      max_num_neighbors=max_num_neighbors,
                      neighbor_cutoff=4)

tasks, datasets, transformers = load_pdbbind(featurizer=acf,
                                             save_dir='.',
                                             data_dir='.',
                                             pocket=True,
                                             reload=False,
                                             set_name='core')

class MyTransformer(dc.trans.Transformer):
    def transform_array(x, y, w, ids):
        kept_rows = x != None
    return x[kept_rows], y[kept_rows], w[kept_rows], ids[kept_rows]

datasets = [d.transform(MyTransformer) for d in datasets]

train, val, test = datasets

atom_types = [6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53., -1.]

def replace_numbers(z):
    np.putmask(z, np.isin(z, atom_types, invert=True), -1)
    return z

for ind, (F_b, y_b, w_b, ids_b) in enumerate(
                    train.iterbatches(1,deterministic=True,
                                        pad_batches=True)):
    batch_size = 12
    idx = 154 // batch_size
    to_include = idx * batch_size
    N = 1600
    N_1 = 100
    N_2 = 1500
    M = 12
    F_b = F_b[0][:to_include]
    num_features = F_b[0][0].shape[1]
    frag1_X_b = np.zeros((to_include, N_1, num_features))
    for j in range(to_include):
        frag1_X_b[j] = F_b[j][0]

    frag2_X_b = np.zeros((to_include, N_2, num_features))
    for k in range(to_include):
        frag2_X_b[k] = F_b[k][3]

    complex_X_b = np.zeros((to_include, N, num_features))
    for l in range(to_include):
        complex_X_b[l] = F_b[l][6]

    frag1_Nbrs = np.zeros((to_include, N_1, M))
    frag1_Z_b = np.zeros((to_include, N_1))
    for i in range(to_include):
        z = replace_numbers(F_b[i][2])
        frag1_Z_b[i] = z
    frag1_Nbrs_Z = np.zeros((to_include, N_1, M))
    for atom in range(N_1):
        for i in range(to_include):
            atom_nbrs = F_b[i][1].get(atom, "")
            frag1_Nbrs[i,
                        atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
                frag1_Nbrs_Z[i, atom, j] = frag1_Z_b[i, atom_j]

    frag2_Nbrs = np.zeros((to_include, N_2, M))
    frag2_Z_b = np.zeros((to_include, N_2))
    for i in range(to_include):
        z = replace_numbers(F_b[i][5])
        frag2_Z_b[i] = z
    frag2_Nbrs_Z = np.zeros((to_include, N_2, M))
    for atom in range(N_2):
        for i in range(to_include):
            atom_nbrs = F_b[i][4].get(atom, "")
            frag2_Nbrs[i,
                        atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
                frag2_Nbrs_Z[i, atom, j] = frag2_Z_b[i, atom_j]

    complex_Nbrs = np.zeros((to_include, N, M))
    complex_Z_b = np.zeros((to_include, N))
    for i in range(to_include):
        z = replace_numbers(F_b[i][8])
        complex_Z_b[i] = z
    complex_Nbrs_Z = np.zeros((to_include, N, M))
    for atom in range(N):
        for i in range(to_include):
            atom_nbrs = F_b[i][7].get(atom, "")
            complex_Nbrs[i, atom, :len(atom_nbrs)] = np.array(
                atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
                complex_Nbrs_Z[i, atom, j] = complex_Z_b[i, atom_j]

inputs = [
    frag1_X_b, frag1_Nbrs, frag1_Nbrs_Z,
    frag2_X_b, frag2_Nbrs, frag2_Nbrs_Z,
    complex_X_b, complex_Nbrs, complex_Nbrs_Z
]

names = [
    'lig_X', 'lig_Nbrs', 'lig_Nbrs_Z',
    'prot_X', 'prot_Nbrs', 'prot_Nbrs_Z',
    'complex_X', 'complex_Nbrs', 'complex_Nbrs_Z'
]

for idx, input in enumerate(inputs):
    np.save('./Atomic_Conv/%s.npy' % (names[idx]), input)
  
np.save('./MyDrive/Atomic_Conv/labels.npy', labels)
