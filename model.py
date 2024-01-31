import torch
from torch import nn
from layer import AtomicConvolution

class AtomicModel(nn.Module):
    def __init__(self, atom_types, radial_params, input_shape):
        super(AtomicModel, self).__init__()
        self.atom_types = atom_types
        self.radial_params = radial_params
        self.input_shape = input_shape

        self.aconv_lig = AtomicConvolution(
            self.atom_types, self.radial_params, self.input_shape[0]
        )
        self.aconv_prot = AtomicConvolution(
            self.atom_types, self.radial_params, self.input_shape[1]
        )
        self.aconv_complex = AtomicConvolution(
            self.atom_types, self.radial_params, self.input_shape[2]
        )

        self.lig_sequential = nn.Sequential(
            nn.Linear(100 * 990, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, 1)
        )

        self.prot_sequential = nn.Sequential(
            nn.Linear(1500 * 990, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, 1)
        )

        self.complex_sequential = nn.Sequential(
            nn.Linear(1600 * 990, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, 1)
        )


    def forward(self, inputs):
        lig_conv = torch.flatten(self.aconv_lig(inputs[0]), start_dim=1)
        prot_conv = torch.flatten(self.aconv_prot(inputs[1]), start_dim=1)
        complex_conv = torch.flatten(self.aconv_complex(inputs[2]), start_dim=1)

        lig_proj = self.lig_sequential(lig_conv)
        prot_proj = self.prot_sequential(prot_conv)
        complex_proj = self.complex_sequential(complex_conv)

        deltaG = complex_proj - prot_proj - lig_proj
        return deltaG
