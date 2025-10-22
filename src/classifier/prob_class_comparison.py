#!/usr/bin/env python3
"""
Compare true class fractions (from RusanenEtAl_synthetic_fractions.csv)
with predicted probabilities (from model output).

"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main(true_csv: Path, pred_csv: Path, outdir: Path, penalty: str):
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading true fractions: {true_csv}")
    true_df = pd.read_csv(true_csv, sep=";")
    print(f"Reading predicted probabilities: {pred_csv}")
    pred_df = pd.read_csv(pred_csv, sep=";")

    # --- Class mapping ---
    class_map = {
        "HOA": "HOA",
        "BBOA": "BBOA",
        "SOAtraffic": "more oxidized",
        "SOAbio": "less oxidized"
    }
    

    # --- Combine true + predicted values ---
    combined_df = pd.DataFrame(index=true_df.index)
    for true_cls, pred_cls in class_map.items():
        combined_df[f"{true_cls}_true"] = true_df[true_cls]
        combined_df[f"{true_cls}_pred"] = pred_df[f"proba:{pred_cls}"]

    # --- Save combined CSV ---
    combined_csv_path = outdir / f"fractions_vs_probabilities_{penalty}.csv"
    combined_df.to_csv(combined_csv_path, index=False, sep=";", encoding="utf-8-sig")

    # --- Scatter plots per class ---
    for cls in class_map.keys():
        plt.figure(figsize=(5, 5))
        plt.scatter(combined_df[f"{cls}_true"], combined_df[f"{cls}_pred"], alpha=0.7)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("True fraction")
        plt.ylabel("Predicted probability")
        plt.title(f"{cls}: True vs Predicted for {penalty}")
        plt.grid(True)
        plt.tight_layout()
        plot_path = outdir / f"scatter_{cls}_{penalty}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare predicted probabilities with true fractions.")
    parser.add_argument("--true_csv", type=Path, required=True)
    parser.add_argument("--pred_csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("comparison_output"))
    parser.add_argument("--penalty", type=str, required=True)
    args = parser.parse_args()
    main(args.true_csv, args.pred_csv, args.outdir, args.penalty)
