import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from Measurements import Measurement   # your class

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run NMF on dataset using Measurements class.")
    parser.add_argument("--input", required=True, help="Path to input .xlsx or .csv file.")
    parser.add_argument("--output", required=True, help="Output prefix for saved F/G matrices.")
    parser.add_argument("--k", type=int, default=5, help="Number of sources/components for NMF.")
    parser.add_argument("--max_iter", type=int, default=500, help="Maximum iterations for NMF.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance for NMF.")
    args = parser.parse_args()

    # -------------------------------------------------------
    # Load using Measurements class
    # -------------------------------------------------------
    meas = Measurement(input_path=args.input, output_prefix=args.output, plot_subdir="NMF")
    meas.load()

    X_np = meas.get_X()              # n × m NumPy
    time = meas.get_time()           # n-vector
    mz_labels = meas.get_mz_labels() # list of length m

    # -------------------------------------------------------
    # Fit NMF
    # -------------------------------------------------------
    model = NMF(n_components=args.k, init='nndsvda', max_iter=args.max_iter, tol=args.tol, random_state=42)
    G_learned = model.fit_transform(X_np)   # n × k
    F_learned = model.components_.T         # m × k
    meas.set_F(F_learned)
    meas.set_G(G_learned)
    # -------------------------------------------------------
    # Save F and G matrices
    # -------------------------------------------------------
    meas.Excel_results_creation()

    print("Saved learned F and G matrices.")

    # -------------------------------------------------------
    # Compare with ground truth if available
    # -------------------------------------------------------
    meas.compare_to_ground_truth()

    # -------------------------------------------------------
    # Plot learned profiles and residuals
    # -------------------------------------------------------
    X_hat = G_learned @ F_learned.T   # Reconstructed X
    meas.plot()
    meas.plot_scaled_residuals(X_hat)
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
