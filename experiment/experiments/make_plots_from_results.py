"""
make_plots_from_results.py
==========================
Load all patched rep_*.npz files from a completed subsampling experiment
and regenerate:
  - PP-plots (null + real data panels, S-GLM vs MUM in different colours)
  - FPR summary bar chart
  - metrics CSV

Usage
-----
python make_plots_from_results.py \
    [--raw_dir  <path>]           # default depends on --use_ukb
    [--plots_dir <path>]          # default: same experiment tree / results/plots
    [--metrics_dir <path>]        # default: same experiment tree / results/metrics
    [--N_list 50 100 200 ...]     # default: all N found in raw_dir
    [--alpha_threshold 0.05]
"""

import argparse
import csv
import gc
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist
import torch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.linewidth": 0.8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))   # …/experiments/
PARENT_DIR = os.path.dirname(EXPERIMENT_DIR)                  # …/experiment/
for _p in (PARENT_DIR, EXPERIMENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from util import preprocess_Z
from run_subsampling_experiment import (
    compute_fpr,
    fit_sglm,
    load_base_data,
    load_ukb_data,
    make_multigroup_data,
    mum_pvalues,
    sglm_pvalues,
    stratified_subsample,
)

DEFAULT_EXPERIMENT_PREFIX_GRF = "subsampling_experiment_GRF_R"
DEFAULT_EXPERIMENT_PREFIX_UKB = "subsampling_experiment_UKB_R"
NEW_KEYS = {
    "p_sglm", "z_sglm", "p_sglm_real", "z_sglm_real",
    "p_mum_real", "z_mum_real",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["plot", "patch", "patch_and_plot"],
                   default="plot")
    p.add_argument("--use_ukb", action="store_true",
                   help="Use UKB defaults instead of GRF defaults when --raw_dir is not provided")
    p.add_argument("--raw_dir",     default=None,
                   help="Input raw results directory (default depends on --use_ukb)")
    p.add_argument("--plots_dir",   default=None,
                   help="Output directory for plots (default: raw_dir/../../plots)")
    p.add_argument("--metrics_dir", default=None,
                   help="Output directory for CSV  (default: raw_dir/../../metrics)")
    p.add_argument("--N_list", nargs="+", type=int, default=None,
                   help="Subset of N values to plot (default: all found)")
    p.add_argument("--alpha_threshold", type=float, default=0.05)
    p.add_argument("--rep", type=int, default=None,
                   help="Process only this repetition index in patch mode")
    p.add_argument("--spacing", type=int, default=5)
    p.add_argument("--polynomial_order", type=int, default=1)
    p.add_argument("--inference_method", default="sandwich",
                   choices=["FI", "sandwich"])
    p.add_argument("--n_age_bins", type=int, default=5)
    p.add_argument("--force", action="store_true",
                   help="Re-compute missing keys even if already present")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def resolve_raw_dir(args) -> str:
    if args.raw_dir is not None:
        return args.raw_dir

    prefix = DEFAULT_EXPERIMENT_PREFIX_UKB if args.use_ukb else DEFAULT_EXPERIMENT_PREFIX_GRF
    candidates = []
    for name in os.listdir(EXPERIMENT_DIR):
        if not name.startswith(prefix):
            continue
        raw_dir = os.path.join(EXPERIMENT_DIR, name, "results", "raw")
        if os.path.isdir(raw_dir):
            candidates.append(raw_dir)

    if not candidates:
        dataset = "UKB" if args.use_ukb else "GRF"
        raise FileNotFoundError(
            f"No default raw results directory found for {dataset}. "
            f"Expected a folder like {prefix}*/results/raw under {EXPERIMENT_DIR}. "
            "Please pass --raw_dir explicitly."
        )

    candidates.sort(key=os.path.getmtime, reverse=True)
    chosen = candidates[0]
    logger.info("Using default raw_dir: %s", chosen)
    return chosen


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(raw_dir: str, N_list) -> list[dict]:
    """Load all rep_*.npz files and return list of result dicts."""
    results = []
    for fname in sorted(os.listdir(raw_dir)):
        if not (fname.startswith("rep_N") and fname.endswith(".npz")):
            continue
        try:
            parts = fname.replace(".npz", "").split("_")
            N    = int(parts[1][1:])
            rep  = int(parts[2][3:])
            seed = int(parts[3][4:])
        except (IndexError, ValueError):
            continue
        if N_list is not None and N not in N_list:
            continue

        d = np.load(os.path.join(raw_dir, fname), allow_pickle=False)
        r = {"N": N, "rep": rep, "seed": seed}
        for key in ("p_sglm", "z_sglm", "p_mum", "z_mum",
                    "p_sglm_real", "z_sglm_real",
                    "p_mum_real",  "z_mum_real",
                    "fpr_sglm", "fpr_mum"):
            if key in d.files:
                val = d[key]
                # scalar arrays → Python float
                r[key] = float(val) if val.ndim == 0 else val
            else:
                r[key] = None
        results.append(r)

    logger.info("Loaded %d rep files from %s", len(results), raw_dir)
    return results


def discover_files(raw_dir: str, N_list, rep: int = None):
    """Return list of (N, rep, seed, path) for all matching rep_*.npz files."""
    entries = []
    for fname in sorted(os.listdir(raw_dir)):
        if not (fname.startswith("rep_N") and fname.endswith(".npz")):
            continue
        try:
            parts = fname.replace(".npz", "").split("_")
            N = int(parts[1][1:])
            r = int(parts[2][3:])
            seed = int(parts[3][4:])
        except (IndexError, ValueError):
            continue
        if N_list is not None and N not in N_list:
            continue
        if rep is not None and r != rep:
            continue
        entries.append((N, r, seed, os.path.join(raw_dir, fname)))
    return entries


def patch_one(entry, base_data, args, device):
    """Patch a single .npz file. Returns (patched: bool, error: str|None)."""
    N, rep, seed, path = entry
    logger.info("Processing: %s", path)

    try:
        existing = dict(np.load(path, allow_pickle=False))
        missing = NEW_KEYS - set(existing.keys())

        if not missing and not args.force:
            logger.info("SKIP  N=%4d rep=%3d — all keys present", N, rep)
            return False, None

        logger.info("PATCH N=%4d rep=%3d seed=%d — missing: %s",
                    N, rep, seed, sorted(missing))

        if args.dry_run:
            return True, None

        rng = np.random.default_rng(seed)
        age_col = base_data.get("age_col", 2)

        Z_raw = base_data["Z"]
        Y_full = base_data["Y"]
        X_spatial = base_data["X_spatial"]

        idx = stratified_subsample(Z_raw, n_bins=args.n_age_bins,
                                   n_sample=N, rng=rng, age_col=age_col)
        Z_sub_raw = Z_raw[idx]
        Y_sub = Y_full[idx]

        Z_sub_pre = preprocess_Z(simulated_dset=False, Z=Z_sub_raw,
                                 polynomial_order=args.polynomial_order)
        age_col_preprocessed = 1

        Z_sub_for_reg = Z_sub_raw[:, 1:]
        data_sub = make_multigroup_data(Z_sub_for_reg, Y_sub, X_spatial)
        data_sub_inf = make_multigroup_data(Z_sub_pre, Y_sub, X_spatial)

        perm_order = rng.permutation(len(idx))
        Z_perm_raw = Z_sub_raw[perm_order]
        Z_perm_pre = preprocess_Z(simulated_dset=False, Z=Z_perm_raw,
                                  polynomial_order=args.polynomial_order)
        Z_perm_for_reg = Z_perm_raw[:, 1:]

        payload = dict(existing)

        if "p_sglm_real" not in existing or args.force:
            p_sr, z_sr = sglm_pvalues(
                data_sub_inf,
                fit_sglm(data_sub, "Poisson", "log", device,
                         simulated_dset=False),
                args.inference_method,
                args.polynomial_order,
                device,
                simulated_dset=False,
                age_col_preprocessed=age_col_preprocessed,
            )
            payload["p_sglm_real"] = p_sr.astype(np.float32)
            payload["z_sglm_real"] = z_sr.astype(np.float32)
            logger.info("  + p_sglm_real computed")

        if "p_mum_real" not in existing or args.force:
            intercept = np.ones((Z_sub_pre.shape[0], 1))
            Z_mum = np.concatenate([Z_sub_pre, intercept], axis=1)
            p_mr, z_mr = mum_pvalues(Z_mum, Y_sub.astype(np.float64),
                                     age_col=age_col_preprocessed)
            payload["p_mum_real"] = p_mr.astype(np.float32)
            payload["z_mum_real"] = z_mr.astype(np.float32)
            logger.info("  + p_mum_real computed")

        if "p_sglm" not in existing or args.force:
            data_perm_reg = make_multigroup_data(Z_perm_for_reg, Y_sub, X_spatial)
            data_perm_inf = make_multigroup_data(Z_perm_pre, Y_sub, X_spatial)
            p_s, z_s = sglm_pvalues(
                data_perm_inf,
                fit_sglm(data_perm_reg, "Poisson", "log", device,
                         simulated_dset=False),
                args.inference_method,
                args.polynomial_order,
                device,
                simulated_dset=False,
                age_col_preprocessed=age_col_preprocessed,
            )
            payload["p_sglm"] = p_s.astype(np.float32)
            payload["z_sglm"] = z_s.astype(np.float32)
            payload["fpr_sglm"] = np.array(
                float(compute_fpr(p_s, args.alpha_threshold))
            )
            logger.info("  + p_sglm (null) computed, FPR=%.4f",
                        float(payload["fpr_sglm"]))

        np.savez_compressed(path, **payload)
        gc.collect()
        return True, None
    except Exception as exc:
        logger.exception("Failed for N=%d rep=%d seed=%d", N, rep, seed)
        return False, str(exc)


# ---------------------------------------------------------------------------
# PP-plot helpers
# ---------------------------------------------------------------------------

def _neglog10_p(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, np.finfo(float).tiny, 1.0)
    return -np.log10(x)


def _mean_rejection_percent(p_list, alpha_threshold: float):
    """Average percent of voxels rejected at the given threshold."""
    if not p_list:
        return None

    rates = []
    for a in p_list:
        a = np.asarray(a).ravel()
        a = a[np.isfinite(a)]
        a = a[(a >= 0) & (a <= 1)]
        if a.size == 0:
            continue
        rates.append(100.0 * float(np.mean(a < alpha_threshold)))

    if not rates:
        return None
    return float(np.mean(rates))


def _add_rejection_text(ax, entries, alpha_threshold: float):
    """Add a small text box summarizing voxel rejection percentages."""
    lines = [rf"Rejected voxels ($p<{alpha_threshold:g}$)"]
    for label, value in entries:
        if value is not None:
            lines.append(f"{label}: {value:.2f}%")

    if len(lines) == 1:
        return

    ax.text(
        0.965,
        0.055,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        linespacing=1.25,
        color="0.20",
        bbox={
            "boxstyle": "round,pad=0.30,rounding_size=0.08",
            "facecolor": "white",
            "edgecolor": "0.85",
            "linewidth": 0.6,
            "alpha": 0.94,
        },
        zorder=10,
    )


def _pp_curve_mean_ci(ax, p_list, colour, label, ci=0.95):
    """Plot mean PP-curve across realisations with pointwise Beta-distribution CI.

    For each rank position k (out of V voxels), under H0 the k-th order
    statistic of V Uniform(0,1) p-values follows Beta(k, V-k+1).  The shaded
    band shows the pointwise [alpha/2, 1-alpha/2] envelope.  The solid line
    shows the mean empirical CDF averaged across R realisations.
    """
    if not p_list:
        return

    arrays = []
    for a in p_list:
        a = np.asarray(a).ravel()
        a = a[np.isfinite(a)]
        a = a[(a >= 0) & (a <= 1)]

        if len(a) > 0:
            arrays.append(np.sort(a))

    if not arrays:
        return

    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        logger.warning(
            "%s: unequal p-value lengths found: %s. Interpolating to common rank grid.",
            label, sorted(set(lengths))
        )

    R = len(arrays)
    V = min(lengths)
    probs = (np.arange(1, V + 1, dtype=float) - 0.5) / V

    mat = np.vstack([
        np.quantile(a, probs, method="linear") if len(a) != V else a
        for a in arrays
    ])

    mean_obs = np.mean(mat, axis=0)

    alpha_ci = 1.0 - ci
    k = np.arange(1, V + 1, dtype=float)
    expected = k / (V + 1.0)
    lo = beta_dist.ppf(alpha_ci / 2.0, k, V - k + 1.0)
    hi = beta_dist.ppf(1.0 - alpha_ci / 2.0, k, V - k + 1.0)

    x_plot = _neglog10_p(expected)
    y_mean = _neglog10_p(mean_obs)
    y_lo = _neglog10_p(hi)
    y_hi = _neglog10_p(lo)

    ax.fill_between(
        x_plot,
        y_lo,
        y_hi,
        color=colour,
        alpha=0.12,
        linewidth=0,
        label="_nolegend_",
        zorder=1,
    )
    ax.plot(
        x_plot,
        y_mean,
        color=colour,
        lw=2.0,
        label=label,
        alpha=0.95,
        solid_capstyle="round",
        zorder=3,
    )

    finite_vals = np.concatenate([
        x_plot[np.isfinite(x_plot)],
        y_mean[np.isfinite(y_mean)],
        y_lo[np.isfinite(y_lo)],
        y_hi[np.isfinite(y_hi)],
    ])
    return float(np.max(finite_vals)) if finite_vals.size else None


def _style_pp_axis(ax, max_val: float, panel_label: str, alpha_line: float,
                   alpha_threshold: float):
    """Apply consistent manuscript-style formatting to one PP-plot axis."""
    ax.plot([0, max_val], [0, max_val], color="0.15", ls="--", lw=1.0,
        label="Uniform", zorder=2)
    ax.axhline(alpha_line, color="0.35", linestyle=":", lw=1.0,
           label=rf"$p={alpha_threshold:g}$")
    ax.axvline(alpha_line, color="0.35", linestyle=":", lw=1.0)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.90", linewidth=0.55)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.25")
    ax.spines["bottom"].set_color("0.25")
    ax.text(
        -0.10,
        1.04,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )


def _clean_pp_legend(ax):
    """Use a compact legend suitable for a two-panel manuscript figure."""
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    clean_handles, clean_labels = [], []
    for handle, label in zip(handles, labels):
        if label == "_nolegend_" or label in seen:
            continue
        seen.add(label)
        clean_handles.append(handle)
        clean_labels.append(label)
    ax.legend(
        clean_handles,
        clean_labels,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="0.85",
        facecolor="white",
        fontsize=7,
        handlelength=1.8,
        handletextpad=0.45,
        borderpad=0.30,
        labelspacing=0.30,
        loc="upper left",
        bbox_to_anchor=(0.045, 0.955))


def make_pp_plot(results: list, N: int, plots_dir: str, ts: str,
                 R_label: int = None, alpha_threshold: float = 0.05):
    """Two-panel PP-plot for a given N: null (left) and real data (right)."""
    p_sglm_null, p_mum_null = [], []
    p_sglm_real, p_mum_real = [], []
    alpha_line = _neglog10_p(alpha_threshold)
    colours = {
        "sglm": "#D55E00",  # vermillion, colourblind-safe
        "mum": "#0072B2",   # blue, colourblind-safe
    }

    for r in results:
        if r["N"] != N:
            continue
        if r.get("p_sglm") is not None:
            p_sglm_null.append(r["p_sglm"])
        if r.get("p_mum") is not None:
            p_mum_null.append(r["p_mum"])
        if r.get("p_sglm_real") is not None:
            p_sglm_real.append(r["p_sglm_real"])
        if r.get("p_mum_real") is not None:
            p_mum_real.append(r["p_mum_real"])

    if not any([p_sglm_null, p_mum_null, p_sglm_real, p_mum_real]):
        logger.warning("N=%d: no p-value arrays found, skipping PP-plot", N)
        return

    R = R_label or len([r for r in results if r["N"] == N])
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.45), constrained_layout=True)

    # Left: null (permuted age)
    ax = axes[0]
    null_rej_sglm = _mean_rejection_percent(p_sglm_null, alpha_threshold)
    null_rej_mum = _mean_rejection_percent(p_mum_null, alpha_threshold)
    max_null = max(
        [0.0] + [
            v for v in [
                _pp_curve_mean_ci(ax, p_sglm_null, colours["sglm"], "S-GLM"),
                _pp_curve_mean_ci(ax, p_mum_null,  colours["mum"], "MUM"),
            ]
            if v is not None
        ]
    )
    max_null = max(1.0, max_null, alpha_line) * 1.02
    _style_pp_axis(ax, max_null, "A", alpha_line, alpha_threshold)
    ax.set_title(f"Permuted-age null\n$N={N}$, $R={R}$", pad=7)
    _clean_pp_legend(ax)
    _add_rejection_text(ax, [("S-GLM", null_rej_sglm), ("MUM", null_rej_mum)],
                        alpha_threshold)

    # Right: real data
    ax = axes[1]
    real_rej_sglm = _mean_rejection_percent(p_sglm_real, alpha_threshold)
    real_rej_mum = _mean_rejection_percent(p_mum_real, alpha_threshold)
    max_real = max(
        [0.0] + [
            v for v in [
                _pp_curve_mean_ci(ax, p_sglm_real, colours["sglm"], "S-GLM"),
                _pp_curve_mean_ci(ax, p_mum_real,  colours["mum"], "MUM"),
            ]
            if v is not None
        ]
    )
    max_real = max(1.0, max_real, alpha_line) * 1.02
    _style_pp_axis(ax, max_real, "B", alpha_line, alpha_threshold)
    ax.set_title(f"Observed UKB data\n$N={N}$, $R={R}$", pad=7)
    _clean_pp_legend(ax)
    _add_rejection_text(ax, [("S-GLM", real_rej_sglm), ("MUM", real_rej_mum)],
                        alpha_threshold)

    out = os.path.join(plots_dir, f"PPplot_N{N}_{ts}.png")
    out_pdf = os.path.join(plots_dir, f"PPplot_N{N}_{ts}.pdf")
    fig.savefig(out)
    fig.savefig(out_pdf)
    plt.close(fig)
    logger.info("Saved PP-plot (N=%d) → %s and %s", N, out, out_pdf)


# ---------------------------------------------------------------------------
# FPR summary
# ---------------------------------------------------------------------------

def aggregate_metrics(results: list, alpha: float) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        N = r["N"]
        for method in ("sglm", "mum"):
            fpr = r.get(f"fpr_{method}")
            if fpr is not None and not np.isnan(fpr):
                grouped[N][method].append(fpr)
    rows = []
    for N in sorted(grouped):
        for method, fprs in grouped[N].items():
            rows.append({
                "N":        N,
                "method":   method.upper(),
                "fpr_mean": float(np.mean(fprs)),
                "fpr_std":  float(np.std(fprs)),
                "fpr_min":  float(np.min(fprs)),
                "fpr_max":  float(np.max(fprs)),
                "n_reps":   len(fprs),
            })
    return rows


def make_fpr_summary_plot(rows: list, alpha: float, plots_dir: str, ts: str):
    if not rows:
        return
    N_vals  = sorted(set(r["N"] for r in rows))
    methods = ["SGLM", "MUM"]
    colours = {"SGLM": "#E64646", "MUM": "#4682B4"}
    x       = np.arange(len(N_vals))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, method in enumerate(methods):
        means, stds = [], []
        for N in N_vals:
            row = next((r for r in rows if r["N"] == N and r["method"] == method), None)
            means.append(row["fpr_mean"] if row else float("nan"))
            stds.append(row["fpr_std"]   if row else 0.0)
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
               label=method, color=colours[method], alpha=0.8)

    ax.axhline(alpha, color="k", linestyle="--", lw=1, label=f"α={alpha}")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(n) for n in N_vals])
    ax.set_xlabel("Subsample size N")
    ax.set_ylabel("Mean FPR")
    ax.set_title("FPR under null (age permuted) — S-GLM vs MUM")
    ax.legend()
    fig.tight_layout()

    out = os.path.join(plots_dir, f"FPR_summary_{ts}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved FPR summary → %s", out)


def save_metrics_csv(rows: list, metrics_dir: str, ts: str):
    if not rows:
        return
    out = os.path.join(metrics_dir, f"metrics_{ts}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved metrics CSV → %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_output_dirs(args):
    base = os.path.dirname(os.path.dirname(args.raw_dir))  # …/results → experiment root
    plots_dir   = args.plots_dir   or os.path.join(base, "results", "plots")
    metrics_dir = args.metrics_dir or os.path.join(base, "results", "metrics")
    os.makedirs(plots_dir,   exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    return plots_dir, metrics_dir


def run_patch_mode(args):
    entries = discover_files(args.raw_dir, args.N_list, rep=args.rep)
    n_vals = sorted(set(e[0] for e in entries))
    logger.info("Found %d files to process in %s", len(entries), args.raw_dir)
    logger.info("N_list: %s", n_vals)
    if args.rep is not None:
        logger.info("Single-rep mode: rep=%d", args.rep)
    if not entries:
        logger.error("No result files found in %s", args.raw_dir)
        return 1

    if args.use_ukb:
        logger.info("Loading UKB data (spacing=%d)…", args.spacing)
        base_data = load_ukb_data(spacing=args.spacing)
        logger.info("UKB data: n=%d, V=%d",
                    base_data["Z"].shape[0], base_data["Y"].shape[1])
    else:
        logger.info("Loading GRF data (n_subject=100, seed=0)…")
        base_data = load_base_data(n_subject=100, data_seed=0)
        logger.info("GRF data: n=%d, V=%d",
                    base_data["Z"].shape[0], base_data["Y"].shape[1])

    device = torch.device("cpu")
    n_patched = 0
    n_errors = 0
    for entry in entries:
        patched, err = patch_one(entry, base_data, args, device)
        if err:
            n_errors += 1
        elif patched:
            n_patched += 1

    logger.info("Done. Patched=%d  Skipped=%d  Errors=%d",
                n_patched, len(entries) - n_patched - n_errors, n_errors)
    return 0 if n_errors == 0 else 1


def run_plot_mode(args):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir, metrics_dir = resolve_output_dirs(args)

    results = load_results(args.raw_dir, args.N_list)
    if not results:
        logger.error("No result files found in %s", args.raw_dir)
        return 1

    N_vals = sorted(set(r["N"] for r in results))
    logger.info("N values found: %s", N_vals)

    # Report key availability
    for key in ("p_sglm", "p_mum", "p_sglm_real", "p_mum_real"):
        n_have = sum(1 for r in results if r.get(key) is not None)
        logger.info("  %-16s : %d / %d reps have this key",
                    key, n_have, len(results))

    # PP-plots per N
    for N in N_vals:
        R = sum(1 for r in results if r["N"] == N)
        make_pp_plot(results, N, plots_dir, ts, R_label=R,
                     alpha_threshold=args.alpha_threshold)

    # FPR summary
    rows = aggregate_metrics(results, args.alpha_threshold)
    make_fpr_summary_plot(rows, args.alpha_threshold, plots_dir, ts)
    save_metrics_csv(rows, metrics_dir, ts)

    # Print table
    logger.info("\n%s", "-" * 62)
    logger.info("%-8s  %-8s  %10s  %10s  %6s", "N", "Method",
                "FPR mean", "FPR std", "n_reps")
    logger.info("%s", "-" * 62)
    for row in rows:
        logger.info("%-8d  %-8s  %10.4f  %10.4f  %6d",
                    row["N"], row["method"],
                    row["fpr_mean"], row["fpr_std"], row["n_reps"])
    logger.info("%s", "-" * 62)
    logger.info("Plots saved to: %s", plots_dir)
    return 0


def main():
    args = get_args()
    args.raw_dir = resolve_raw_dir(args)

    if args.mode == "patch":
        sys.exit(run_patch_mode(args))

    if args.mode == "patch_and_plot":
        status = run_patch_mode(args)
        if status != 0:
            sys.exit(status)

    sys.exit(run_plot_mode(args))


if __name__ == "__main__":
    main()
