#!/usr/bin/env python3
"""
Create a normalized class-fraction CSV from RusanenEtAl_synthetic.xlsx (sheet 'G').

"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main(xlsx_path: Path, outdir: Path):
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    print(f"Reading {xlsx_path} (sheet 'G')...")
    df = pd.read_excel(xlsx_path, sheet_name="G")

    class_cols = df.select_dtypes(include="number").columns.tolist()
    time_col = "time/source" if "time/source" in df.columns else None
    if time_col and time_col in class_cols:
        class_cols.remove(time_col)

    if not class_cols:
        raise ValueError("No numeric columns found to treat as classes.")

    print(f"Detected class columns: {class_cols}")

    # Compute normalized class fractions
    totals = df[class_cols].sum(axis=1)
    fractions = df[class_cols].div(totals, axis=0).fillna(0)

    # Add time/source or index column
    if time_col:
        fractions.insert(0, time_col, df[time_col])
    else:
        fractions.insert(0, "index", range(len(df)))

    # Save output CSV
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "RusanenEtAl_synthetic_fractions.csv"
    fractions.to_csv(out_path, index=False, sep=";", encoding="utf-8-sig")
    
    print(f"\nSaved class fractions{out_path}")
    print(fractions.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a normalized fraction CSV from RusanenEtAl_synthetic.xlsx (sheet 'G')."
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=Path("RusanenEtAl_synthetic.xlsx"),
        help="Path to the Excel file (default: RusanenEtAl_synthetic.xlsx)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("resources"),
        help="Output directory to save the CSV (default: resources)",
    )
    args = parser.parse_args()
    main(args.xlsx, args.outdir)
