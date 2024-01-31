import torch
from torch import nn

class AtomicConvolution(nn.Module):
    def __init__(self, atom_types, radial_params, input_shape):
        super(AtomicConvolution, self).__init__()
        self.atom_types = torch.tensor(atom_types, dtype=torch.int32)
        self.radial_params = radial_params
        self.input_shape = input_shape

        vars = []
        for i in range(3):
            val = np.array([p[i] for p in self.radial_params]).reshape(
                (-1, 1, 1, 1))
            vars.append(torch.tensor(val, dtype=torch.float32))

        self.rc = nn.Parameter(vars[0])
        self.rs = nn.Parameter(vars[1])
        self.re = nn.Parameter(vars[2])

        self.batch_norm = nn.BatchNorm1d(self.input_shape)

    def distance_tensor(self, X, Nbrs, B, N, M, d):
        flat_neighbors = Nbrs.view((-1, N * M))
        neighbor_coords = torch.zeros((B, N * M, d))
        for _ in range(B):
            neighbor_coords[_] = X[_, flat_neighbors[_], ...]
        D = neighbor_coords.view([-1, N, M, d]) - X.unsqueeze(2)
        return D

    def distance_matrix(self, D):
        R = torch.sum((D * D), dim=3)
        return torch.sqrt(R)

    def gaussian_distance_matrix(self, R, rs, e):
        return torch.exp(-e * (R - rs) ** 2)

    def radial_cutoff(self, R, rc):
        PI = torch.acos(torch.zeros(1)).item() * 2
        T = 0.5 * (torch.cos(PI * R / rc) + 1)
        E = torch.zeros_like(T)
        cond = torch.le(R, rc)
        FC = torch.where(cond, T, E)
        return FC

    def radial_symmetry_function(self, R, rc, rs, e):
        K = self.gaussian_distance_matrix(R, rs, e)
        FC = self.radial_cutoff(R, rc)
        return K * FC

    def forward(self, inputs):
        X, Nbrs, Nbrs_Z = inputs
        N = X.shape[-2]
        M = Nbrs.shape[-1]
        B = X.shape[0]
        d = X.shape[-1]

        D = self.distance_tensor(X, Nbrs, B, N, M, d)
        R = self.distance_matrix(D).unsqueeze(0)
        rsf = self.radial_symmetry_function(R, self.rc, self.rs, self.re)

        sym = []
        length = len(self.atom_types)
        for i in range(length):
            cond = (Nbrs_Z == self.atom_types[i]).float().view((1, -1, N, M))
            sym.append(torch.sum(cond * rsf, dim=3))
        layer = torch.cat(sym, dim=0).permute((1, 2, 0))
        return self.batch_norm(layer)
