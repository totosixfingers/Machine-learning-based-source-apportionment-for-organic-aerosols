import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

from Measurements import Measurement
from Autoencoders import Autoencoder
from Autoencoders import SourceBasedAE

# -----------------------------------------------------------
# Training Function
# -----------------------------------------------------------


def nndsvd_init(X, k):
    """
    NNDSVDa initialization using scikit-learn's NMF.
    X: (n_samples, n_features)
    Returns W_init (k, n_features), H_init (n_features, k)
    """
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    model = NMF(
        n_components=k,
        init="nndsvda",
        max_iter=300,        # full optimization, not just 1 iter
        random_state=0,
        solver="cd"
    )
    W = model.fit_transform(X_np)           # shape: (n_samples, k)
    H = model.components_                   # shape: (k, n_features)

    # Construct weight tensors
    W_init = torch.tensor(H, dtype=torch.float32)   # (k, n_features)
    H_init = torch.tensor(H.T, dtype=torch.float32)  # (n_features, k)
    return W_init, H_init


def compute_loss(X, X_hat, E, eps=1e-6):
    """
    mean( ((X - X_hat) / E_safe)**2 )
    where E_safe is clipped for stability.
    """
    # Convert error matrix to tensor if needed
    if isinstance(E, np.ndarray):
        E = torch.tensor(E, dtype=torch.float32, device=X.device)

    # Clip uncertainties
    # Avoid too small sigmas (explode gradients)
    # Avoid too large sigmas (loss becomes too small)
    E_safe = torch.clamp(E, min=eps, max=torch.quantile(E, 0.99))

    R = (X - X_hat) / E_safe
    return torch.mean(R * R)


def train(X, E, k=5, lr=1e-2, epochs=500):
    n_samples, m = X.shape
    model = Autoencoder(m, k)

    # Initialize with NNDSVD
    W_init, H_init = nndsvd_init(X, k)
    model.encoder.weight.data = W_init       # shape: (k, m)
    model.decoder.weight.data = H_init     # shape: (m, k)

    '''print("\nInitial weights of standard Autoencoder:")
    print("Encoder weights:\n", model.encoder.weight.data)
    print("Decoder weights:\n", model.decoder.weight.data)'''
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        X_hat, Z = model(X)
        loss = compute_loss(X, X_hat, E)
        loss.backward()
        optimizer.step()
        
        # Enforce non-negativity in decoder/encoder weights
        with torch.no_grad():
            model.encoder.weight.data.clamp_(min=0)
            model.decoder.weight.data.clamp_(min=0)

        Z = model.normalize_F_and_rescale_G(Z)
        
        losses.append(loss.item())

    print(f"Finished Training | Final Loss = {loss.item():.6f}")
    return model, losses


def train_source_based(X, E, k=5, lr=1e-2, epochs=500,
                       F_fixed=None, n_fixed=0, allow_scale_fixed=True):
    n_samples, m = X.shape

    # Initialize the SourceBasedAE
    model = SourceBasedAE(m, k, F_fixed=F_fixed, n_fixed=n_fixed,
                          allow_scale_fixed=allow_scale_fixed)

    F_free_init, _ = nndsvd_init(X, model.n_free)

    # assign to your model
    F_free_init = F_free_init   # shape: (m, n_free)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        X_hat, G, F = model(X)
        loss = compute_loss(X, X_hat, E)
        loss.backward()
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN detected at epoch", epoch)
            break

        optimizer.step()
        model.clamp_nonneg()  # ensures F_free stays non-negative
        
        G = model.normalize_profiles(G)
        losses.append(loss.item())

    print(f"Finished Training | Final Loss = {loss.item():.6f}")
    return model, losses


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run standard and source-based autoencoder on dataset.")
    parser.add_argument("--input", required=True,
                        help="Path to input .xlsx or .csv file.")
    parser.add_argument("--output", required=True,
                        help="Output prefix for saved F/G matrices.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--fixed_profiles", required=True,
                        help="Path to .xlsx containing rows labeled HOA, COA, BBOA")
    parser.add_argument(
        "--fixed_labels", nargs="+", default=["HOA", "CCOA", "BBOA"],
        help="Fixed profile labels to use (space-separated). Example: HOA CCOA BBOA"
    )
    parser.add_argument(
        "--n_runs_sourceae",
        type=int,
        default=1,
        help="Number of repeated runs for Source-Based Autoencoder."
    )

    parser.add_argument(
        "--random_fixed_profiles",
        action="store_true",
        help="If set, re-sample fixed profiles from library on each SourceAE run."
    )
    args = parser.parse_args()

    # -------------------------------------------------------
    # Load data
    # -------------------------------------------------------
    meas_ae = Measurement(input_path=args.input,
                          output_prefix=args.output, plot_subdir="AE")
    meas_ae.load()
    X_ae = torch.tensor(meas_ae.get_X(), dtype=torch.float32)

    meas_Source_ae = Measurement(input_path=args.input, F_fixed_path=args.fixed_profiles,
                                 output_prefix=args.output, plot_subdir="SourceAE")
    meas_Source_ae.load()

    # =======================================================
    # 1. Standard Autoencoder
    # =======================================================
    print("\nTraining standard Autoencoder...")
    E_ae = torch.tensor(meas_ae.get_error(), dtype=torch.float32)
    model_ae, losses_ae = train(
        X_ae, E_ae, k=args.k, lr=args.lr, epochs=args.epochs)

    F_ae = model_ae.decoder.weight.data.cpu().numpy()
    G_ae = model_ae.encoder(X_ae).detach().cpu().numpy()
    X_hat_ae, _ = model_ae(X_ae)

    meas_ae.set_F(F_ae)
    meas_ae.set_G(G_ae)

    # Save CSVs
    meas_ae.Excel_results_creation()
    print("Saved standard AE F/G matrices.")

    # Compare, plot, residuals
    meas_ae.compare_to_ground_truth()
    meas_ae.plot()
    meas_ae.plot_scaled_residuals(X_hat_ae.detach().numpy())

    # =======================================================
    # 2. Source-Based Autoencoder
    # =======================================================
    X_Sae = torch.tensor(meas_Source_ae.get_X(), dtype=torch.float32)
    # meas_Source_ae.get_F_Truth() #for ground truth rus
    n_runs = args.n_runs_sourceae if args.n_runs_sourceae > 0 else 1
    F_runs = []
    G_runs = []
     # 2

    for run_idx in range(n_runs):
        print(f"\nTraining Source-Based Autoencoder (run {run_idx+1}/{n_runs})...")
        E_sae = torch.tensor(meas_ae.get_error(), dtype=torch.float32)
        
        meas_Source_ae.load_fixed_profiles(
            labels=tuple(args.fixed_labels),
            random_fixed=args.random_fixed_profiles
        )
            
        F_fixed = meas_Source_ae.get_F_fixed()
        n_fixed = meas_Source_ae.get_n_fixed() 
        
        model_sbae, losses_sbae = train_source_based(
        X_Sae, E_sae, k=args.k, lr=args.lr, epochs=args.epochs,
        F_fixed=F_fixed, n_fixed=n_fixed, allow_scale_fixed=True
        )

        F_curr = model_sbae.build_F().detach().cpu().numpy()
        G_curr = model_sbae.encoder(X_Sae).detach().cpu().numpy()
        
        F_runs.append(F_curr)
        G_runs.append(G_curr)
        
    F_runs = np.stack(F_runs, axis=0)
    G_runs = np.stack(G_runs, axis=0)
    
    # Simple average across runs (for now)
    F_sbae_mean = F_runs.mean(axis=0)
    G_sbae_mean = G_runs.mean(axis=0)

    # std for later uncertainty plots:
    F_sbae_std = F_runs.std(axis=0)
    G_sbae_std = G_runs.std(axis=0)
    X_hat_sbae = G_sbae_mean @ F_sbae_mean.T

    meas_Source_ae.set_F(F_sbae_mean)
    meas_Source_ae.set_G(G_sbae_mean)

    # Save CSVs
    meas_Source_ae.Excel_results_creation()

    # Compare, plot, residuals
    meas_Source_ae.plot_uncertainty(F_runs, G_runs)
    meas_Source_ae.compare_to_ground_truth()
    meas_Source_ae.plot()
    meas_Source_ae.plot_scaled_residuals(X_hat_sbae)


# -----------------------------------------------------------
if __name__ == "__main__":
    main()
