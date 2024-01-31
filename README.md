This is a PyTorch implementation of the atomic convolutional network for predicting protein/ligand binding affinities. The link to the paper: https://arxiv.org/abs/1703.10603.
I have also utilized bits of the DeepChem (i.e. the featurizer class and have taken some inspiration form their AtomicConvolution layer class): https://github.com/deepchem/deepchem/blob/master/deepchem/models/atomic_conv.py.

I found that using their featurizer saves a lot of time, although you can implement a variation of it using their code or use other methods to do so. You could preprocess your data
using VMD and then have it report all the neighboring atoms for specific atom types. 

A big limitation to this is that it is very memory intensive. That is due to the number of atoms used, ligands typically have less but proteins can have quite a lot, plus representing it as a complex.
