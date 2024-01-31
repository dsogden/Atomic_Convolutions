This is a PyTorch implementation of the atomic convolutional network for predicting protein/ligand binding affinities. The link to the paper: https://arxiv.org/abs/1703.10603.
I have also utilized bits from DeepChems methods/classes (i.e. the featurizer class and have taken some inspiration from their AtomicConvolution layer class): https://github.com/deepchem/deepchem/blob/master/deepchem/models/atomic_conv.py.

I found that using their featurizer saves a lot of time, although you can implement a variation of it using their code or use other methods to do so. For example, you could preprocess your data using VMD and then tailor a number of bash scripts to handle all the preprocessing. 

A big limitation to this model is that it is very memory intensive, as the number of atoms used for the protein/protein-ligand complex can get rather high.
