import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from Measurements import Measurement
from Autoencoders import Autoencoder
from Autoencoders import SourceBasedAE

# -----------------------------------------------------------
# Training Function
# -----------------------------------------------------------


def train(X, k=5, lr=1e-2, epochs=500):
    n_samples, m = X.shape
    model = Autoencoder(m, k)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        X_hat, Z = model(X)
        loss = criterion(X_hat, X)
        loss.backward()
        optimizer.step()

        # Enforce non-negativity in decoder/encoder weights
        with torch.no_grad():
            model.encoder.weight.data.clamp_(min=0)
            model.decoder.weight.data.clamp_(min=0)

        losses.append(loss.item())

    print(f"Finished Training | Final Loss = {loss.item():.6f}")
    return model, losses


def train_source_based(X, k=5, lr=1e-2, epochs=500,
                       F_fixed=None, n_fixed=0, allow_scale_fixed=True):
    n_samples, m = X.shape

    # Initialize the SourceBasedAE
    model = SourceBasedAE(m, k, F_fixed=F_fixed, n_fixed=n_fixed,
                          allow_scale_fixed=allow_scale_fixed)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        X_hat, G, F = model(X)
        loss = criterion(X_hat, X)
        loss.backward()
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN detected at epoch", epoch)
            break

        optimizer.step()
        model.clamp_nonneg()  # <-- ensures F_free stays non-negative
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
    meas_Source_ae.load_fixed_profiles()

    # =======================================================
    # 1. Standard Autoencoder
    # =======================================================
    '''print("\nTraining standard Autoencoder...")
    model_ae, losses_ae = train(X_ae, k=args.k, lr=args.lr, epochs=args.epochs)

    F_ae = model_ae.decoder.weight.data.cpu().numpy()
    G_ae = model_ae.encoder(X_ae).detach().cpu().numpy()
    X_hat_ae, _ = model_ae(X_ae)

    meas_ae.set_F(F_ae)
    meas_ae.set_G(G_ae)

    # Save CSVs
    pd.DataFrame(F_ae, index=meas_ae.get_mz_labels(), columns=[f"Source_{i+1}" for i in range(F_ae.shape[1])])\
        .to_csv(args.output + "_F_AE.csv")
    pd.DataFrame(G_ae, index=meas_ae.get_time(), columns=[f"Source_{i+1}" for i in range(G_ae.shape[1])])\
        .to_csv(args.output + "_G_AE.csv")
    print("Saved standard AE F/G matrices.")

    # Compare, plot, residuals
    meas_ae.compare_to_ground_truth()
    meas_ae.plot()
    meas_ae.plot_scaled_residuals(X_hat_ae.detach().numpy())'''

    # =======================================================
    # 2. Source-Based Autoencoder
    # =======================================================
    X_Sae = torch.tensor(meas_Source_ae.get_X(), dtype=torch.float32)
    F_fixed = meas_Source_ae.get_F_fixed() #meas_Source_ae.get_F_Truth() #for ground truth rus
    n_fixed = meas_Source_ae.get_n_fixed() # 2

    print("\nTraining Source-Based Autoencoder...")
    model_sbae, losses_sbae = train_source_based(
        X_Sae, k=args.k, lr=args.lr, epochs=args.epochs,
        F_fixed=F_fixed, n_fixed=n_fixed, allow_scale_fixed=True
    )

    F_sbae = model_sbae.build_F().detach().cpu().numpy()
    G_sbae = model_sbae.encoder(X_Sae).detach().cpu().numpy()
    X_hat_sbae, _, _ = model_sbae(X_Sae)

    meas_Source_ae.set_F(F_sbae)
    meas_Source_ae.set_G(G_sbae)

    # Save CSVs
    pd.DataFrame(F_sbae, index=meas_Source_ae.get_mz_labels(), columns=[f"Source_{i+1}" for i in range(F_sbae.shape[1])])\
        .to_csv(args.output + "_F_SourceAE.csv")
    pd.DataFrame(G_sbae, index=meas_Source_ae.get_time(), columns=[f"Source_{i+1}" for i in range(G_sbae.shape[1])])\
        .to_csv(args.output + "_G_SourceAE.csv")
    print("Saved source-based AE F/G matrices.")

    # Compare, plot, residuals
    meas_Source_ae.compare_to_ground_truth()
    meas_Source_ae.plot()
    meas_Source_ae.plot_scaled_residuals(X_hat_sbae.detach().numpy())


# -----------------------------------------------------------
if __name__ == "__main__":
    main()
