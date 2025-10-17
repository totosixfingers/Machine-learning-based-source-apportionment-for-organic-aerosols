#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from logger import ConsoleLogger


def to_int_if_numeric(col):
    """Convert a column name that looks numeric (e.g., '29') to int (29)."""
    try:
        return int(col)
    except (ValueError, TypeError):
        return col


def load_model_and_features(model_path: str) -> Tuple[object, Optional[List[int]]]:
    """
    Load joblib. Prefer dict {"model": pipeline, "used_mz": [...] }.
    Fallback: if it's a bare pipeline, return (pipeline, None).
    """
    import joblib

    obj = joblib.load(model_path)
    
    #print(obj.keys())
    #print(obj["label_encoder"].classes_)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        used_mz = obj.get("used_mz")
        le = obj.get("label_encoder")
        return model, used_mz, le
    return obj, None,None  # bare pipeline fallback


def pick_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Return a likely time column name if present (e.g., 'time/m:z').
    We keep it for the output; it's ignored as a feature.
    """
    candidates = ["time/m:z", "time", "t", "index"]
    for c in candidates:
        if c in df.columns:
            return c
    first = df.columns[0]
    if not isinstance(first, (int, np.integer)):
        return first
    return None


def align_features(
    df: pd.DataFrame,
    used_mz: Iterable[int],
    time_col: Optional[str] = None,
    show_max: int = 10,
) -> pd.DataFrame:
    """
    Ensure df has exactly the used_mz columns in that order.
    Missing columns are zero-filled; extra columns are ignored.
    Also surfaces where present columns have NaN values.
    """
    log = ConsoleLogger()

    df = df.copy()
    df.columns = [to_int_if_numeric(c) for c in df.columns]

    # Drop obvious non-feature columns
    drop_cols = {"label", "where"}
    if time_col:
        drop_cols.add(time_col)

    present = {
        c for c in df.columns if isinstance(c, (int, np.integer)) and c not in drop_cols
    }

    # Report completely missing columns (absent from CSV)
    missing_cols = [mz for mz in used_mz if mz not in present]
    if missing_cols:
        log.warning(f"Missing required m/z columns (absent from CSV): {missing_cols}")
        log.warning("   → These columns are zero-filled for all rows.")

    # Report extra columns in the CSV (ignored)
    extra_cols = sorted(present - set(used_mz))
    if extra_cols:
        log.info(f"Ignoring extra m/z columns: {extra_cols}")

    # For columns that DO exist, surface where values are NaN
    nan_info = {}
    for mz in used_mz:
        if mz in present:
            nan_idx = df.index[df[mz].isna()].tolist()
            if nan_idx:
                if time_col and time_col in df.columns:
                    # show row_index:time_value pairs for quick pinpointing
                    examples = [f"{i}:{df.at[i, time_col]}" for i in nan_idx[:show_max]]
                else:
                    examples = nan_idx[:show_max]
                nan_info[mz] = (len(nan_idx), examples)

    if nan_info:
        log.warning("Missing values (NaN) detected in present m/z columns:")
        for mz, (count, examples) in nan_info.items():
            example_str = ", ".join(map(str, examples))
            tail = " …" if count > show_max else ""
            if time_col and time_col in df.columns:
                log.warning(
                    f"   m/z {mz}: {count} rows (row:time examples: {example_str}{tail})"
                )
            else:
                log.warning(
                    f"   m/z {mz}: {count} rows (row indices: {example_str}{tail})"
                )

    # Build the feature matrix in the training order, zero-filling missing columns
    out_cols = []
    for mz in used_mz:
        if mz in present:
            out_cols.append(df[mz])
        else:
            out_cols.append(pd.Series(0.0, index=df.index, name=mz))

    X = pd.concat(out_cols, axis=1)
    X.columns = list(used_mz)
    return X


def row_fraction_normalize(X: pd.DataFrame) -> pd.DataFrame:
    """
    Row-wise fractional normalization (sum to 1). Treat NaN as 0; protect zero rows.
    """
    X = X.fillna(0.0)
    sums = X.sum(axis=1).replace(0, np.nan)
    X_frac = X.div(sums, axis=0).fillna(0.0)
    return X_frac


def _final_estimator(model):
    """Try to retrieve the final estimator from a pipeline-like object."""
    try:
        if hasattr(model, "named_steps") and model.named_steps:
            return list(model.named_steps.values())[-1]
        if hasattr(model, "steps") and model.steps:
            return model.steps[-1][1]
    except Exception:
        pass
    return model


def compute_probabilities(model, X: np.ndarray) -> Tuple[pd.DataFrame, List]:
    """Always return probabilities as a DataFrame; raise if not supported."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes_ = getattr(
            _final_estimator(model), "classes_", getattr(model, "classes_", None)
        )
        if classes_ is None:
            raise AttributeError("Could not locate classes_ on the model.")
        proba_cols = [f"proba:{str(c)}" for c in classes_]
        return pd.DataFrame(proba, columns=proba_cols), list(classes_)
    raise AttributeError("The loaded model does not support predict_proba().")


def make_output_filename(model_path: str, outdir: Path) -> Path:
    """predictions_[model_name].csv inside the given outdir."""
    stem = Path(model_path).stem
    return outdir / f"predictions_{stem}.csv"


def make_meta_filename(model_path: str, outdir: Path) -> Path:
    """predictions_[model_name].meta.txt sidecar next to the CSV."""
    stem = Path(model_path).stem
    return outdir / f"predictions_{stem}.meta.txt"


def main(input_csv: str, model_path: str, outdir: str):
    logger = ConsoleLogger()

    # Ensure output directory exists
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load model
    model, used_mz, le = load_model_and_features(model_path)
    

    # 2) Load data
    df = pd.read_csv(input_csv)
    df.columns = [to_int_if_numeric(c) for c in df.columns]
    time_col = pick_time_column(df)

    # 3) Figure out features to use
    if used_mz is None:
        raise ValueError(
            "Model file lacks 'used_mz'. Re-train with the updated trainer so the "
            "feature list is embedded in the .joblib."
        )

    # 4) Build feature matrix matching training layout
    X_df = align_features(df, used_mz=used_mz, time_col=time_col)
    X_frac = row_fraction_normalize(X_df)
    X = X_frac.values

    # 5) Predict + probabilities
    y_pred_int = model.predict(X)
    proba = model.predict_proba(X)

    # Map numeric predictions to human-readable class names
    if le is not None:  # if LabelEncoder exists in the saved model
        y_pred = le.inverse_transform(y_pred_int)  # map integers back to original names
        proba_cols = [f"proba:{cls}" for cls in le.classes_]
        classes_ = list(le.classes_)
        
    else:
        y_pred = y_pred_int
        proba_cols = [f"proba:{i}" for i in range(proba.shape[1])]
        classes_ = [str(i) for i in range(proba.shape[1])]

    proba_df = pd.DataFrame(proba, columns=proba_cols)


# Compute absolute (mass-weighted) contributions if total measured mass is available
    mz_cols = [c for c in df.columns if isinstance(c, (int, np.integer))]
    if mz_cols:
        total_mass = df[mz_cols].sum(axis=1)
        abs_mass_df = proba_df.mul(total_mass, axis=0)
        # Rename to use the same class names as in proba_df
        abs_mass_df.columns = [c.replace("proba:", "mass:") for c in proba_df.columns]
    else:
        abs_mass_df = pd.DataFrame(index=df.index)  # empty if no m/z columns

    # 6) Compose output table
    out = pd.DataFrame({"pred_label": y_pred}, index=df.index)
    if time_col and time_col in df.columns:
        out.insert(0, time_col, df[time_col])

    # include only probabilities and absolute (mass-weighted) contributions
    out = pd.concat([out, proba_df, abs_mass_df], axis=1)



    # 7) Save CSV and a tiny sidecar .meta.txt
    out_path = make_output_filename(model_path, outdir)
    model_name = Path(model_path).name
    out.to_csv(out_path, index=False)

    meta_path = make_meta_filename(model_path, outdir)
    with open(meta_path, "w", encoding="utf-8") as mf:
        mf.write(f"model: {model_name}\n")

    logger.info(f"Saved predictions (with probabilities) to {out_path}")
    logger.info(f"Saved metadata to {meta_path}")
    logger.info(f"Classes: {', '.join(map(str, classes_))}")
    
    # 8) some statistics
    
    mean_probs = proba_df.mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    mean_probs.plot(kind="bar", ax=ax)
    ax.set_title(f"Mean predicted probabilities ({Path(model_path).stem})")
    ax.set_ylabel("Mean probability")
    ax.set_xlabel("Class")
    plt.tight_layout()
    prob_plot_path = outdir / f"mean_probabilities_{Path(model_path).stem}.png"
    plt.savefig(prob_plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved mean probability plot → {prob_plot_path}")
    
    if mz_cols:
        predicted_total_mass = abs_mass_df.sum(axis=1)
        measured_total_mass = total_mass

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(measured_total_mass, predicted_total_mass, alpha=0.6)
        minv, maxv = (
            min(measured_total_mass.min(), predicted_total_mass.min()),
            max(measured_total_mass.max(), predicted_total_mass.max()),
        )
        ax.plot([minv, maxv], [minv, maxv], "r--", label="1:1 line")
        ax.set_title(f"Mass consistency check ({Path(model_path).stem})")
        ax.set_xlabel("Measured total mass")
        ax.set_ylabel("Predicted total mass")
        ax.legend()
        plt.tight_layout()
        mass_plot_path = outdir / f"mass_consistency_{Path(model_path).stem}.png"
        plt.savefig(mass_plot_path, dpi=200)
        plt.close(fig)
        logger.info(f"Saved total mass consistency plot → {mass_plot_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to X_sheet.csv")
    ap.add_argument("--model", required=True, help="Path to the saved model .joblib")
    ap.add_argument(
        "--outdir",
        default=".",
        help="Folder where the prediction file will be saved (default: current directory)",
    )
    args = ap.parse_args()
    main(args.csv, args.model, args.outdir)
