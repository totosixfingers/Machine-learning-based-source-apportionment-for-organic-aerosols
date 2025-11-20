import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


# -----------------------------------------------------------
# Autoencoder Models
# -----------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, m, k):
        super().__init__()
        self.encoder = nn.Linear(m, k, bias=False)
        self.decoder = nn.Linear(k, m, bias=False)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class SourceBasedAE(nn.Module):
    def __init__(self, m, k, F_fixed=None, n_fixed=0, allow_scale_fixed=True):
        super().__init__()
        self.m, self.k = m, k
        self.n_fixed = int(n_fixed)
        self.n_free = k - self.n_fixed

        # Encoder
        self.encoder = nn.Linear(m, k, bias=False)

        # Fixed sources
        if self.n_fixed > 0 and F_fixed is not None:
            F_fixed_t = torch.tensor(F_fixed, dtype=torch.float32)
            F_fixed_t = F_fixed_t#[:, :self.n_fixed] #only for ground truth 
            
            #L1 normalization: avoid division by zero
            #col_sum = F_fixed_t.sum(dim=0, keepdim=True)
            #col_sum[col_sum == 0] = 1.0  # for NaN not sure if needed
            #F_fixed_t /= col_sum
    
            self.register_buffer("F_fixed", F_fixed_t)
            
        else:
            self.register_buffer("F_fixed", torch.empty((m, 0)))
           

        # Free sources
        if self.n_free > 0:
            self.F_free = nn.Parameter(0.01 * torch.rand((m, self.n_free)))
        else:
            self.F_free = None

    def build_F(self):
        parts = []
        if self.n_fixed > 0:
            parts.append(self.F_fixed)
        if self.n_free > 0:
            parts.append(self.F_free)
        return torch.cat(parts, dim=1) if parts else torch.zeros((self.m, self.k))

    def forward(self, X):
        G = self.encoder(X)
        F = self.build_F()
        X_hat = G @ F.T
        return X_hat, G, F

    def clamp_nonneg(self):
        if self.F_free is not None:
            self.F_free.data.clamp_(min=0)
        