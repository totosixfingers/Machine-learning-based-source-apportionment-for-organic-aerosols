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
    
    def normalize_F_and_rescale_G(self, Z):
        """
        Normalize the columns of F (decoder weights)
        and rescale Z accordingly so that reconstruction ZF^T stays unchanged.
        """
        with torch.no_grad():
            F = self.decoder.weight.data           # (m, k)
            colsum = F.sum(dim=0, keepdim=True)    # (1, k)
            colsum = torch.clamp(colsum, min=1e-12)

            # Normalize F
            self.decoder.weight.data = F / colsum

            # Rescale contributions Z (shape batch × k)
            Z *= colsum
        return Z


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
            self.scale_fixed = nn.Parameter(torch.ones(self.n_fixed))
            
            self.register_buffer("F_fixed", F_fixed_t)
            
        else:
            self.register_buffer("F_fixed", torch.empty((m, 0)))
            self.scale_fixed = None
           

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
    
    def normalize_profiles(self, G):
        """
        Normalize ALL F columns and rescale G so that X_hat remains unchanged.
        This enforces sum(F[:, j]) = 1 for each source j.
        """
        with torch.no_grad():
            F = self.build_F()   # shape (m, k)

            # Compute column sums
            colsum = F.sum(dim=0)          # (k,)
            colsum_safe = torch.clamp(colsum, min=1e-12)

            # ------------------------
            # Rescale FIXED profiles
            # ------------------------
            if self.n_fixed > 0:
                # Apply current scale to fixed profiles
                F_fixed_scaled = self.F_fixed * self.scale_fixed

                # Normalize fixed profiles
                norm_fixed = colsum_safe[:self.n_fixed]
                self.scale_fixed.data /= norm_fixed

            # ------------------------
            # Normalize FREE profiles
            # ------------------------
            if self.n_free > 0:
                norm_free = colsum_safe[self.n_fixed:].unsqueeze(0)   # (1, n_free)
                self.F_free.data /= norm_free

            # ------------------------
            # Rescale G accordingly
            # ------------------------
            G *= colsum_safe.unsqueeze(0)   # shape (1,k) → broadcast to (n_samples, k)

        return G

            