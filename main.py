import itertools
from itertools import product as iterprod
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from model import AtomicModel
from utils import load_data, train_step, eval_step

path = './Atomic_Conv'
names = [
    'lig_X', 'lig_Nbrs', 'lig_Nbrs_Z',
    'prot_X', 'prot_Nbrs', 'prot_Nbrs_Z',
    'complex_X', 'complex_Nbrs', 'complex_Nbrs_Z'
]

X_1, Nbrs_1, Nbrs_Z_1, X_2, Nbrs_2, Nbrs_Z_2, X_complex, Nbrs_complex, Nbrs_Z_complex, labels = load_data(path, names)
X_1, Nbrs_1, Nbrs_Z_1 = torch.tensor(X_1, dtype=torch.float32), torch.tensor(Nbrs_1, dtype=torch.int32), torch.tensor(Nbrs_Z_1, dtype=torch.int32)
X_2, Nbrs_2, Nbrs_Z_2 = torch.tensor(X_2, dtype=torch.float32), torch.tensor(Nbrs_2, dtype=torch.int32), torch.tensor(Nbrs_Z_2, dtype=torch.int32)
X_complex, Nbrs_complex, Nbrs_Z_complex = torch.tensor(X_complex, dtype=torch.float32), torch.tensor(Nbrs_complex, dtype=torch.int32), torch.tensor(Nbrs_Z_complex, dtype=torch.int32)
labels = torch.tensor(labels, dtype=torch.float32)

atom_types = [6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53., -1.]
radial = [
    [
        1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
        8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
    ],
    [0.0, 4.0, 8.0],
    [0.4]
]

rp = [x for x in iterprod(*radial)]
input_shapes = [100, 1500, 1600]
model = AtomicModel(atom_types, rp, input_shapes)

EPOCHS = 25
LR = 1E-4
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

batch_size = 12
idx = 154 // batch_size

X_1, Nbrs_1, Nbrs_Z_1 = X_1.view(idx, batch_size, 100, 3), Nbrs_1.view(idx, batch_size, 100, 12), Nbrs_Z_1.view(idx, batch_size, 100, 12)
X_2, Nbrs_2, Nbrs_Z_2 = X_2.view(idx, batch_size, 1500, 3), Nbrs_2.view(idx, batch_size, 1500, 12), Nbrs_Z_2.view(idx, batch_size, 1500, 12)
X_complex, Nbrs_complex, Nbrs_Z_complex = X_complex.view(idx, batch_size, 1600, 3), Nbrs_complex.view(idx, batch_size, 1600, 12), Nbrs_Z_complex.view(idx, batch_size, 1600, 12)

labels = labels[:idx * batch_size].view(idx, batch_size, 1)

def main():
    indices = torch.arange(12)
    train_indices = indices[:9]
    val_indices = indices[-3:]
    
    training_loss = np.zeros((EPOCHS, 3))
    
    for epoch in tqdm(range(EPOCHS)):
        train_batch_loss = 0
        val_batch_loss = 0
    
        model.train()
        for i in train_indices:
            inputs = [
                [X_1[i], Nbrs_1[i], Nbrs_Z_1[i]],
                [X_2[i], Nbrs_2[i], Nbrs_Z_2[i]],
                [X_complex[i], Nbrs_complex[i], Nbrs_Z_complex[i]]
            ]
            y = labels[i]
            train_loss = train_step(inputs, y, model, criterion, optimizer)
            train_batch_loss += train_loss
    
        model.eval()
        for i in val_indices:
            inputs = [
                [X_1[i], Nbrs_1[i], Nbrs_Z_1[i]],
                [X_2[i], Nbrs_2[i], Nbrs_Z_2[i]],
                [X_complex[i], Nbrs_complex[i], Nbrs_Z_complex[i]]
            ]
            y = labels[i]
            val_loss = eval_step(inputs, y, model)
            val_batch_loss += train_loss
    
        train_batch_loss /= 9
        val_batch_loss /= 3
        training_loss[epoch] = np.array([epoch + 1, train_batch_loss, val_batch_loss])
    
        print(f'Epoch: {epoch + 1}, Training Loss = {train_batch_loss} Validation Loss = {val_batch_loss}')
    
        perm = torch.randperm(9)
        train_indices = train_indices[perm]
        perm = torch.randperm(3)
        val_indices = val_indices[perm]
