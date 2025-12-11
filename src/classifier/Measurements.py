import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class Measurement:
    """
    Handles:
      - Loading measurement data (X, time)
      - Loading ground-truth F/G when available
      - Managing output paths (CSV, plots)
      - Plotting autoencoder results
      - Comparing learned vs ground-truth F/G
    """

    def __init__(self, input_path, F_fixed_path=None, X=None, time=None, mz_labels=None,
                 output_prefix="output", plot_subdir="plots"):
        # Store paths
        self.input_path = input_path
        self.output_prefix = output_prefix

        # Create plot directory specific for this instance
        self.plot_dir = os.path.join(self.output_prefix, plot_subdir)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Output CSVs
        self.output_F = os.path.join(
            self.plot_dir, f"{plot_subdir}_F_learned.csv")
        self.output_G = os.path.join(
            self.plot_dir, f"{plot_subdir}_G_learned.csv")
        self.output_RMSE = os.path.join(
            self.plot_dir, f"{plot_subdir}_RMSE.csv")

        # Plot files
        self.plot_F = os.path.join(
            self.plot_dir, f"F_profiles_{plot_subdir}.png")
        self.plot_G = os.path.join(
            self.plot_dir, f"G_profiles_{plot_subdir}.png")
        self.plot_stacked = os.path.join(
            self.plot_dir, f"stacked_contributions_{plot_subdir}.png")
        self.plot_corr = os.path.join(
            self.plot_dir, f"correlation_G_{plot_subdir}.png")

        # Data placeholders
        self.X = X
        self.time = time
        self.mz_labels = mz_labels
        self.F_fixed_path = F_fixed_path
        self.F_fixed = None
        self.n_fixed = None
        self.F_truth = None
        self.G_truth = None
        self.F_learned = None
        self.G_learned = None
        self.E = None

    # ---------------------------------------------------------
    #                     DATA LOADING
    # ---------------------------------------------------------

    def load(self):
        """Loads X, time, mz labels, and optional ground-truth F/G."""

        xls = pd.ExcelFile(self.input_path)
        sheets = xls.sheet_names
        print(f"Available sheets: {sheets}")

        # Load X
        if "measurements" in sheets:
            df_X = pd.read_excel(self.input_path, sheet_name="measurements")
        elif "X" in sheets:
            df_X = pd.read_excel(self.input_path, sheet_name="X")
        else:
            raise ValueError("Could not find 'measurements' or 'X' sheet.")

        # First column = time or Date
        # self.time = df_X.iloc[:, 0].values
        self.time = pd.to_datetime(df_X.iloc[:, 0], dayfirst=True)

        # Remaining columns = measurement matrix X
        self.X = df_X.iloc[:, 1:].replace(
            ",", ".", regex=True).astype(float).values

        # m/z labels = column names
        # Extract numeric m/z values safely from column names
        self.mz_labels = []
        for col in df_X.columns[1:]:
            # Use regex to extract numeric part
            match = re.search(r"[-+]?\d*\.\d+|\d+", str(col))
            if match:
                self.mz_labels.append(float(match.group()))
            else:
                raise ValueError(
                    f"Could not parse numeric m/z from column '{col}'")

        # Convert to numpy array
        self.mz_labels = np.array(self.mz_labels)

        # Load ground truth if present
        if "F" in sheets:
            df_F = pd.read_excel(self.input_path, sheet_name="F")
            self.F_truth = df_F.iloc[:, 1:].values

        if "G" in sheets:
            df_G = pd.read_excel(self.input_path, sheet_name="G")
            self.G_truth = df_G.iloc[:, 1:].values

        print("Data loaded successfully.")

        # ----- error sheet -----
        if "error" in sheets:
            df_err = pd.read_excel(self.input_path, sheet_name="error")
            # assume same structure as X: first column time, rest = errors
            self.E = df_err.iloc[:, 1:].replace(
                ",", ".", regex=True).astype(float).values
        else:
            self.E = None

    # ---------------------------------------------------------
    #                     GROUND TRUTH CHECK
    # ---------------------------------------------------------
    def has_ground_truth(self):
        return (self.F_truth is not None) and (self.G_truth is not None)

    # ---------------------------------------------------------
    #             COMPARE LEARNED VS GROUND TRUTH
    # ---------------------------------------------------------
    def compare_to_ground_truth(self):
        if not self.has_ground_truth():
            print("No ground truth available. Skipping comparison.")
            return
        F_truth = self.F_truth
        G_truth = self.G_truth

        n_sources = self.G_learned.shape[1]
        matched_indices = []

        print("\nComparison with Ground Truth (max correlation matching):")

        for i in range(n_sources):
            # Compute correlations of learned source i with all ground truth sources
            corr_Gs = [np.corrcoef(self.G_learned[:, i], G_truth[:, j])[
                0, 1] for j in range(n_sources)]
            corr_Fs = [np.corrcoef(self.F_learned[:, i], F_truth[:, j])[
                0, 1] for j in range(n_sources)]

            # Find ground truth source with maximum correlation
            max_corr_G_idx = int(np.argmax(np.abs(corr_Gs)))
            max_corr_F_idx = int(np.argmax(np.abs(corr_Fs)))

            matched_indices.append((max_corr_G_idx, max_corr_F_idx))

            # RMSE with the matched ground truth source
            rmse_G = np.sqrt(
                np.mean((self.G_learned[:, i] - G_truth[:, max_corr_G_idx]) ** 2))
            rmse_F = np.sqrt(
                np.mean((self.F_learned[:, i] - F_truth[:, max_corr_F_idx]) ** 2))

            print(
                f"Learned Source {i+1}: "
                f"matches Ground Truth G Source {max_corr_G_idx+1} | "
                f"G corr = {corr_Gs[max_corr_G_idx]:.3f}, G RMSE = {rmse_G:.3f} | "
                f"matches Ground Truth F Source {max_corr_F_idx+1} | "
                f"F corr = {corr_Fs[max_corr_F_idx]:.3f}, F RMSE = {rmse_F:.3f}"
            )
    # ---------------------------------------------------------
    #                         PLOTTING
    # ---------------------------------------------------------

    def plot(self):
        """
        Saves:
            - F profiles
            - G profiles
            - stacked G contributions
            - G correlation scatterplots
        """
        if not self.has_ground_truth():
            print("No ground truth available. Skipping comparison.")
            return

        F_truth = self.F_truth
        G_truth = self.G_truth
        time = self.time
        mz = np.arange(self.F_learned.shape[0])
        k = self.F_learned.shape[1]

        # ------------------ 1. F profiles ----------------------
        fig, axs = plt.subplots(k, 1, figsize=(10, 2 * k), sharex=True)

        width = 0.35  # bar width offset for truth bars

        for i in range(k):
            # Learned
            axs[i].bar(self.mz_labels,
                       self.F_learned[:, i],
                       width=width,
                       label=f"Source {i+1}",
                       color="C0")

            # Truth overlay
            if F_truth is not None:
                axs[i].bar(self.mz_labels + width,
                           F_truth[:, i],
                           width=width,
                           label="Truth",
                           color="C1",
                           alpha=0.7)

            axs[i].legend()
            axs[i].set_ylabel("Intensity")

        axs[-1].set_xlabel("m/z")

        # Show all m/z labels
        formatted_labels = [
            str(int(mz)) if float(mz).is_integer() else f"{mz:.2f}"
            for mz in self.mz_labels
        ]

        axs[-1].set_xticks(self.mz_labels)
        axs[-1].set_xticklabels(formatted_labels, fontsize=4, rotation=90)

        plt.tight_layout()
        plt.savefig(self.plot_F, dpi=300)
        plt.close()

        # ------------------ 2. G profiles ----------------------
        fig, axs = plt.subplots(k, 1, figsize=(10, 2*k), sharex=True)
        for i in range(k):
            axs[i].plot(time, self.G_learned[:, i],
                        label=f"Source {i+1}", color="C0")
            if G_truth is not None:
                axs[i].plot(time, G_truth[:, i], "--",
                            color="C1", label="Truth")
            axs[i].legend()
            axs[i].set_ylabel("Contribution")
        axs[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.savefig(self.plot_G, dpi=300)
        plt.close()

        # ------------------ 3. Stacked G -----------------------
        plt.figure(figsize=(10, 4))
        plt.stackplot(time, self.G_learned.T, labels=[
                      f"S{i+1}" for i in range(k)])
        plt.xlabel("Time")
        plt.ylabel("Contribution")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(self.plot_stacked, dpi=300)
        plt.close()

        # ------------------ 4. Correlation ----------------------
        if G_truth is not None:
            fig, axs = plt.subplots(1, k, figsize=(4*k, 4))
            for i in range(k):
                axs[i].scatter(G_truth[:, i], self.G_learned[:, i], alpha=0.5)
                corr = np.corrcoef(G_truth[:, i], self.G_learned[:, i])[0, 1]
                axs[i].set_title(f"Source {i+1}\nr={corr:.2f}")
                axs[i].set_xlabel("Truth")
                axs[i].set_ylabel("Learned")
            plt.tight_layout()
            plt.savefig(self.plot_corr, dpi=300)
            plt.close()

        print(f"\nPlots saved to directory: {self.plot_dir}")

    # ---------------------------------------------------------
    #                RESIDUAL DIAGNOSTIC PLOTS
    # ---------------------------------------------------------
    def plot_scaled_residuals(self, X_hat):
        """
        Creates:
            (a) scaled residuals vs m/z
            (b) scaled residuals vs time
            (c) histogram of all scaled residuals
        Saves plots into the plot directory.
        """

        # Compute scaled residuals
        self.X = np.asarray(self.X)
        X_hat = np.asarray(X_hat)

        eps = 1e-9
        R = np.abs((self.X - X_hat) / (np.maximum(self.X, eps)))

        # File paths
        path_res_mz = os.path.join(self.plot_dir, "residuals_over_mz.png")
        path_res_time = os.path.join(self.plot_dir, "residuals_over_time.png")
        path_res_hist = os.path.join(self.plot_dir, "residuals_histogram.png")

        mz = self.mz_labels

        time = self.time

        # ---------------------------------------------------------
        # (a) Scaled residuals over m/z (mean ± std across time) (add the true mz values)
        # ---------------------------------------------------------
        mean_mz = np.mean(R, axis=0)
        std_mz = np.std(R, axis=0)

        plt.figure(figsize=(10, 4))
        plt.bar(mz, mean_mz, yerr=std_mz, alpha=0.8, capsize=3)
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("m/z")
        plt.ylabel("Scaled residual")
        plt.title("Scaled Residuals Over m/z (Signed, Bar Plot)")

        # Format m/z labels: remove .0 if integer
        formatted_labels = [str(int(float(mz))) if float(
            mz).is_integer() else str(mz) for mz in self.mz_labels]

        # Set x-ticks with small font, no rotation
        plt.xticks(self.mz_labels, formatted_labels, fontsize=4)  # small font
        plt.tight_layout()
        plt.savefig(path_res_mz, dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # (b) Scaled residuals over time (do it over day)
        # ---------------------------------------------------------
        mean_t = np.mean(R, axis=1)
        std_t = np.std(R, axis=1)

        plt.figure(figsize=(10, 4))
        plt.plot(time, mean_t, label="Mean residual")
        plt.fill_between(time, mean_t - std_t, mean_t + std_t, alpha=0.3)
        plt.xlabel("Time")
        plt.ylabel("Scaled residual")
        plt.title("Scaled Residuals Over Time")
        plt.tight_layout()
        plt.savefig(path_res_time, dpi=300)
        plt.close()

        # ----------------------------
        # (c) Histogram of scaled residuals
        # ----------------------------
        R = (self.X - X_hat) / (self.E + eps)     # <-- correct denominator, signed
        flat = R.flatten()

        plt.figure(figsize=(7, 5))

        # Symmetric trimming around 1st–99th percentiles
        p_low, p_high = np.percentile(flat, [1, 99])
        mask = (flat >= p_low) & (flat <= p_high)
        flat_trim = flat[mask]

        # Histogram
        plt.hist(flat_trim, bins=60, density=True, alpha=0.55, color="C0",
                edgecolor="black", linewidth=0.4, label="Residuals (1–99th percentile)")

        # Fit normal distribution
        mu, sigma = flat_trim.mean(), flat_trim.std()
        x = np.linspace(flat_trim.min(), flat_trim.max(), 300)

        plt.plot(x, norm.pdf(x, mu, sigma),
                color="C1", linewidth=2, alpha=0.6, label=f"Normal Fit")

        # KDE
        try:
            kde = gaussian_kde(flat_trim)
            plt.plot(x, kde(x), color="C2", linewidth=2, alpha=0.6, label="KDE")
        except Exception:
            pass

        plt.xlabel("Scaled residual (signed)")
        plt.ylabel("Density")
        plt.title(f"Residual Distribution (μ={mu:.3f}, σ={sigma:.3f})")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_res_hist, dpi=300)
        plt.close()

        print("\nResidual diagnostic plots saved:")
        print(f"  - {path_res_mz}")
        print(f"  - {path_res_time}")
        print(f"  - {path_res_hist}")

    def load_fixed_profiles(self, labels=("HOA", "CCOA", "BBOA"), random_fixed=False, rng_seed=42):
        """
    Load fixed source profiles from library.

    Parameters
    ----------
    labels : tuple of str
        Profile base labels, e.g. ("HOA", "BBOA", ...).
    random_fixed : bool
        If False: deterministic selection (same variants every time per rng_seed).
        If True : random selection each call (used for multi-run uncertainty).
    rng_seed : int
        Seed used when random_fixed=False, for reproducibility.
    """
        print("Loading fixed source profiles (random selection per profile)...")

        df = pd.read_excel(self.F_fixed_path, sheet_name=1, decimal=",")

        df_filtered = df[df["m/z"].isin(self.mz_labels)]
        if random_fixed:
            rng = np.random.default_rng()       # new random state each call
        else:
            rng = np.random.default_rng(rng_seed)
        fixed_profiles = []

        # For each requested profile name
        for label in labels:
            matching_cols = [c for c in df.columns if c.startswith(label)]
            if not matching_cols:
                raise ValueError(f"No columns found for label '{label}'")

            # Randomly select one variant
            selected_col = rng.choice(matching_cols)

            profile = pd.to_numeric(
                df_filtered[selected_col], errors='coerce').to_numpy()
            profile = np.nan_to_num(profile, nan=0.0)

            fixed_profiles.append(profile)
            print(
                f"Selected random variant for {label}: column '{selected_col}'")

        # Stack → shape (m, k_fixed)
        F_fixed = np.stack(fixed_profiles, axis=1)

        print(
            f"Final fixed profile matrix shape: {F_fixed.shape} (m × k_fixed)")
        n_fixed = F_fixed.shape[1]
        F_fixed = np.nan_to_num(F_fixed, nan=0.0)
        self.F_fixed = F_fixed
        self.n_fixed = n_fixed

    def Excel_results_creation(self):
        # Save F with ground truth (when available)
        if self.has_ground_truth():
            F_combined = {}
            for i in range(self.F_learned.shape[1]):
                F_combined[f"Learned_{i+1}"] = self.F_learned[:, i]
                F_combined[f"Truth_{i+1}"] = self.F_truth[:, i]
            F_df = pd.DataFrame(F_combined, index=self.get_mz_labels())
        else:
            F_df = pd.DataFrame(self.F_learned, index=self.get_mz_labels(),
                                columns=[f"Learned_{i+1}" for i in range(self.F_learned.shape[1])])

        F_df.to_csv(self.output_F)

        # Save G with ground truth (when available)
        if self.has_ground_truth():
            G_combined = {}
            for i in range(self.G_learned.shape[1]):
                G_combined[f"Learned_{i+1}"] = self.G_learned[:, i]
                G_combined[f"Truth_{i+1}"] = self.G_truth[:, i]
            G_df = pd.DataFrame(G_combined, index=self.get_time())
        else:
            G_df = pd.DataFrame(self.G_learned, index=self.get_time(),
                                columns=[f"Learned_{i+1}" for i in range(self.G_learned.shape[1])])

        G_df.to_csv(self.output_G)

        if self.has_ground_truth():
            mse_F = mean_squared_error(self.F_truth, self.F_learned)
            mse_G = mean_squared_error(self.G_truth, self.G_learned)

            pd.DataFrame({
                "MSE_F": [mse_F],
                "MSE_G": [mse_G],
            }).to_csv(self.output_RMSE, index=False)

            print("Saved MSE file.")

    def plot_uncertainty(self, F_runs, G_runs):
        """
        Creates uncertainty plots:
        - Shaded confidence intervals for F (profile uncertainty)
        - Shaded confidence intervals for G (time series uncertainty)
        - Histogram of source contributions across runs
        """

    

        save_dir = self.plot_dir

        F_runs = np.array(F_runs)      # shape (n_runs, m, k)
        G_runs = np.array(G_runs)      # shape (n_runs, n_samples, k)

        n_runs, m, k = F_runs.shape
        _, n_samples, _ = G_runs.shape

        # ---------------------------------------------------------
        # 1️⃣ PROFILE UNCERTAINTY (F)
        # ---------------------------------------------------------
        fig, axes = plt.subplots(k, 1, figsize=(10, 2*k), sharex=True)

        for i in range(k):
            F_mean = F_runs[:, :, i].mean(axis=0)
            F_std  = F_runs[:, :, i].std(axis=0)

            axes[i].plot(self.mz_labels, F_mean, color="C0", label=f"Mean Source {i+1}")
            axes[i].fill_between(self.mz_labels,
                                F_mean - F_std,
                                F_mean + F_std,
                                alpha=0.3, color="C0", label="±1σ")
            axes[i].set_ylabel("Intensity")
            axes[i].legend()

        axes[-1].set_xlabel("m/z")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "uncertainty_F.png"), dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # 2️⃣ TIME SERIES UNCERTAINTY (G)
        # ---------------------------------------------------------
        fig, axes = plt.subplots(k, 1, figsize=(12, 2*k), sharex=True)

        for i in range(k):
            G_mean = G_runs[:, :, i].mean(axis=0)
            G_std  = G_runs[:, :, i].std(axis=0)

            axes[i].plot(self.time, G_mean, color="C1", label=f"Mean Source {i+1}")
            axes[i].fill_between(self.time,
                                G_mean - G_std,
                                G_mean + G_std,
                                alpha=0.3, color="C1", label="±1σ")
            axes[i].legend()
            axes[i].set_ylabel("Contribution")

        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "uncertainty_G.png"), dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # 3️⃣ HISTOGRAM OF RUN-TO-RUN VARIATION (per source)
        # ---------------------------------------------------------
        fig, axes = plt.subplots(1, k, figsize=(4*k, 4))

        for i in range(k):
            # Total mass across time per run
            total_per_run = G_runs[:, :, i].sum(axis=1)
            axes[i].hist(total_per_run, bins=25, alpha=0.7, color="C2")
            axes[i].set_title(f"Source {i+1}\nContribution variability")
            axes[i].set_xlabel("Total contribution")
            axes[i].set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "uncertainty_histograms.png"), dpi=300)
        plt.close()

        print("Uncertainty plots saved in:", save_dir)
    
    def get_input_path(self): return self.input_path
    def set_input_path(self, p): self.input_path = p

    def get_output_F(self): return self.output_F
    def set_output_F(self, p): self.output_F = p

    def get_output_G(self): return self.output_G
    def set_output_G(self, p): self.output_G = p

    def get_plot_F(self): return self.plot_F
    def get_plot_G(self): return self.plot_G
    def get_plot_stacked(self): return self.plot_stacked
    def get_plot_corr(self): return self.plot_corr

    def get_X(self): return self.X
    def get_time(self): return self.time
    def get_mz_labels(self): return self.mz_labels
    def get_F_truth(self): return self.F_truth
    def get_G_truth(self): return self.G_truth

    def set_F(self, F): self.F_learned = F
    def get_F(self): return self.F_learned
    def set_G(self, G): self.G_learned = G
    def get_G(self): return self.G_learned

    def set_S(self, S_fixed): self.S_fixed = S_fixed
    def get_S(self): return self.S_fixed
    def set_F_fixed(self, F_fixed): self.F_fixed = F_fixed
    def get_F_fixed(self): return self.F_fixed
    def set_n_fixed(self, n_fixed): self.n_fixed = n_fixed
    def get_n_fixed(self): return self.n_fixed

    def has_error(self):
        return self.E is not None

    def get_error(self):
        return self.E
