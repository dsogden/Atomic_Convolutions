import numpy as np
import torch

def load_data(path, names):
    X_1, Nbrs_1, Nbrs_Z_1 = np.load('%s.npy' % (path + names[0])), np.load('%s.npy' % (path + names[1])), np.load('%s.npy' % (path + names[2]))
    X_2, Nbrs_2, Nbrs_Z_2 = np.load('%s.npy' % (path + names[3])), np.load('%s.npy' % (path + names[4])), np.load('%s.npy' % (path + names[5]))
    X_complex, Nbrs_complex, Nbrs_Z_complex = np.load('%s.npy' % (path + names[6])), np.load('%s.npy' % (path + names[7])), np.load('%s.npy' % (path + names[8]))
    labels = np.load('%s/labels.npy' % (path))

    return X_1, Nbrs_1, Nbrs_Z_1, X_2, Nbrs_2, Nbrs_Z_2, X_complex, Nbrs_complex, Nbrs_Z_complex, labels

def train_step(inputs, y, model, loss, optimizer):
    outputs = model(inputs)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def eval_step(inputs, y, model):
    outputs = model(inputs)
    loss = criterion(outputs, y)

    return loss.item()
