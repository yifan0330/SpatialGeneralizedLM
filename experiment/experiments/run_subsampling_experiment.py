"""
Subsampling + Permutation FPR Analysis: S-GLM vs MUM
=====================================================
Evaluates false positive rate (FPR) for the AGE covariate under a permutation
null across different subsample sizes.

Usage
-----
python run_subsampling_experiment_corrected.py \
    --N_list 50 100 200 \
    --R 50 \
    --inference_method sandwich \
    --base_seed 42

Does NOT modify any existing code or result folders.

Notes on permutation scheme
---------------------------
By default the script uses a simple permutation of the entire raw Z matrix. This
is a valid null for a marginal age effect in the one-covariate GRF setting. For
multi-covariate datasets such as UKB, this breaks the link between all covariates
and Y, so it is closer to a global-null test than a conditional-null test on age.

For an approximate age-only null that preserves non-age covariates and shuffles
age locally within age bins, set:

    --permutation_scheme stratified_age

This is still an approximate restricted permutation, not a fully rigorous
conditional-null test.
"""

import argparse
import concurrent.futures
import csv
import functools
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib  # kept for compatibility with existing environment/scripts
import numpy as np
import scipy.stats
import torch

# ---------------------------------------------------------------------------
# Add experiment directory to path so existing modules are importable
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(EXPERIMENT_DIR)
for _p in (PARENT_DIR, EXPERIMENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from regression import BrainRegression_Approximate
from inference import BrainInference_Approximate
from util import preprocess_Z

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRF_DATA_DIR = os.path.join(PARENT_DIR, "data", "brain", "data_Simulation")
MASK_PATH = os.path.join(
    PARENT_DIR,
    "data",
    "brain",
    "smooth_lesion_mask_Simulation.nii.gz",
)

UKB_DATA_DIR = os.path.join(PARENT_DIR, "data", "UKB")
UKB_MASKED_DATA = os.path.join(
    UKB_DATA_DIR,
    "masked_data_RealDataset_spacing_{spacing}.npz",
)

EXPERIMENT_OUT = None
RAW_DIR = None
METRICS_DIR = None
PLOTS_DIR = None
LOGS_DIR = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _setup_output_dirs(R: int, use_ukb: bool = False) -> None:
    """Initialise output directory globals using the resolved R value."""
    global EXPERIMENT_OUT, RAW_DIR, METRICS_DIR, PLOTS_DIR, LOGS_DIR
    dataset_tag = "UKB" if use_ukb else "GRF"
    EXPERIMENT_OUT = os.path.join(
        EXPERIMENT_DIR,
        f"subsampling_experiment_{dataset_tag}_R{R}",
    )
    RAW_DIR = os.path.join(EXPERIMENT_OUT, "results", "raw")
    METRICS_DIR = os.path.join(EXPERIMENT_OUT, "results", "metrics")
    PLOTS_DIR = os.path.join(EXPERIMENT_OUT, "results", "plots")
    LOGS_DIR = os.path.join(EXPERIMENT_OUT, "logs")
    for _d in [RAW_DIR, METRICS_DIR, PLOTS_DIR, LOGS_DIR]:
        os.makedirs(_d, exist_ok=True)


# ===========================================================================
# Argument parsing
# ===========================================================================

def get_args():
    p = argparse.ArgumentParser(description="Subsampling FPR experiment")
    p.add_argument(
        "--N_list",
        nargs="+",
        type=int,
        default=[50, 100, 200, 500, 1000, 2000],
        help="Subsample sizes to evaluate",
    )
    p.add_argument(
        "--R",
        type=int,
        default=100,
        help="Global number of repetitions per subsample size. Ignored when --seed is set.",
    )
    p.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Master random seed used when --seed is not set",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for a single realization. When provided, the script "
            "runs exactly one realization per N and exits. Use --rep to label "
            "the repetition index for output file naming."
        ),
    )
    p.add_argument(
        "--rep",
        type=int,
        default=None,
        help=(
            "Repetition index to run. With --seed, this only labels output files. "
            "Without --seed, this runs exactly one deterministic realization per N "
            "using a seed derived from (base_seed, N, rep)."
        ),
    )
    p.add_argument(
        "--use_ukb",
        action="store_true",
        help="Use the UKB real dataset instead of GRF simulated data",
    )
    p.add_argument(
        "--data_n_subject",
        type=int,
        default=100,
        help="n_subject tag of the GRF data files; ignored when --use_ukb",
    )
    p.add_argument(
        "--data_seed",
        type=int,
        default=0,
        help="Random seed of the source GRF data file; ignored when --use_ukb",
    )
    p.add_argument(
        "--spacing",
        type=int,
        default=5,
        help="B-spline spacing, must match how X_spatial was built",
    )
    p.add_argument(
        "--polynomial_order",
        type=int,
        default=1,
        help="Polynomial order for preprocess_Z; 1 = linear age",
    )
    p.add_argument(
        "--inference_method",
        type=str,
        default="sandwich",
        choices=["FI", "sandwich"],
        help="Variance estimator for S-GLM inference",
    )
    p.add_argument(
        "--sglm_sandwich_meat",
        type=str,
        default="null_cluster",
        choices=["cluster", "iid", "null_cluster", "null_iid"],
        help=(
            "Meat estimator for S-GLM sandwich inference. "
            "'null_cluster'/'null_iid' (default): score-test sandwich — meat uses "
            "residuals r_null = Y - mu_null where mu_null zeros out the contrast "
            "covariate beta block. Ensures E[r_null]=0 under H0 by construction, "
            "eliminating conservative inflation from approximate S-GLM convergence. "
            "'cluster'/'iid': standard sandwich using fitted residuals r = Y - mu_hat."
        ),
    )
    p.add_argument(
        "--sglm_alpha",
        type=float,
        default=0.05,
        help=(
            "Step size for S-GLM regression updates. Larger values converge faster "
            "but can be unstable; 0.05 is a stable default for UKB calibration."
        ),
    )
    p.add_argument(
        "--sglm_max_iter",
        type=int,
        default=2000,
        help="Maximum S-GLM regression iterations; increased to avoid under-converged beta estimates.",
    )
    p.add_argument(
        "--sglm_tol",
        type=float,
        default=1e-6,
        help="S-GLM convergence tolerance, applied to both absolute and relative beta update norms.",
    )
    p.add_argument("--marginal_dist", type=str, default="Poisson")
    p.add_argument("--link_func", type=str, default="log")
    p.add_argument(
        "--alpha_threshold",
        type=float,
        default=0.05,
        help="Significance threshold for FPR computation",
    )
    p.add_argument(
        "--n_age_bins",
        type=int,
        default=5,
        help="Number of age bins for stratification",
    )
    p.add_argument(
        "--fdr",
        action="store_true",
        help="Also compute FPR after BH-FDR correction",
    )
    p.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes. -1 = all CPUs. Use cautiously for UKB.",
    )
    p.add_argument(
        "--mum_min_count",
        type=int,
        default=1,
        help=(
            "Minimum total count per voxel for MUM. Voxels with sum(Y) below "
            "this threshold are returned as NaN p-values."
        ),
    )
    p.add_argument(
        "--mum_min_nonzero",
        type=int,
        default=1,
        help=(
            "Minimum number of subjects with non-zero count per voxel for MUM. "
            "Stricter than --mum_min_count for sparse data."
        ),
    )
    p.add_argument(
        "--permutation_scheme",
        type=str,
        default="simple",
        choices=["simple", "stratified_age"],
        help=(
            "'simple' = global shuffle of all rows of Z. "
            "'stratified_age' = shuffle age column only, within age strata, "
            "holding other covariates fixed."
        ),
    )
    p.add_argument(
        "--save_aggregated_npz",
        action="store_true",
        help=(
            "Also write aggregated raw_N{N}_{ts}.npz files. Off by default to "
            "avoid duplicating disk writes."
        ),
    )
    p.add_argument(
        "--rep_start",
        type=int,
        default=0,
        help="First global rep index to run, inclusive. Ignored when --seed or --rep is set.",
    )
    p.add_argument(
        "--rep_count",
        type=int,
        default=None,
        help=(
            "Number of reps to run on this device. Defaults to --R. The global "
            "rep index is still used for seeding and output filenames."
        ),
    )
    return p.parse_args()


# ===========================================================================
# Data utilities
# ===========================================================================

def load_ukb_data(spacing: int = 5) -> dict:
    """Load UKB masked NPZ and return raw numpy arrays."""
    fname = UKB_MASKED_DATA.format(spacing=spacing)
    if not os.path.exists(fname):
        available = ""
        if os.path.isdir(UKB_DATA_DIR):
            available = "\n".join(os.listdir(UKB_DATA_DIR))
        raise FileNotFoundError(
            f"UKB masked data file not found: {fname}\n"
            f"Expected spacing={spacing}. Available files in {UKB_DATA_DIR}:\n"
            f"{available}"
        )
    raw = np.load(fname, allow_pickle=True)
    return {
        "Z": raw["Z"].astype(np.float64),
        "Y": raw["Y"].astype(np.float32),
        "X_spatial": raw["X_spatial"].astype(np.float64),
        "age_col": 2,
    }


def load_base_data(n_subject: int, data_seed: int) -> dict:
    """Load a GRF simulation NPZ file and return raw numpy arrays."""
    folder = os.path.join(GRF_DATA_DIR, f"GRF_[{n_subject}]")
    fname = os.path.join(folder, f"GRF_[{n_subject}]_random_seed_{data_seed}.npz")
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Data file not found: {fname}\n"
            f"Run data generation first, e.g. run.py --run_data_generation True."
        )
    raw = np.load(fname, allow_pickle=True)
    group_key = list(raw.files)[0]
    group_dict = raw[group_key].item()
    return {
        "Z": group_dict["Z"].astype(np.float64),
        "Y": group_dict["Y"].astype(np.float32),
        "X_spatial": group_dict["X_spatial"].astype(np.float64),
        "age_col": 0,
    }


def make_multigroup_data(
    Z: np.ndarray,
    Y: np.ndarray,
    X_spatial: np.ndarray,
    group_name: str = "Group_1",
) -> dict:
    """Pack arrays into the multi-group NPZ dict format expected by existing classes."""
    inner = {"Z": Z, "Y": Y, "X_spatial": X_spatial}
    return {group_name: np.array(inner, dtype=object)}


# ===========================================================================
# Stratification utilities
# ===========================================================================

def age_strata(Z_raw: np.ndarray, n_bins: int, age_col: int = 0) -> np.ndarray:
    """Return integer stratum labels for each subject using quantile age bins."""
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    age = Z_raw[:, age_col]
    bins = np.quantile(age, np.linspace(0.0, 1.0, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) <= 2:
        return np.zeros(Z_raw.shape[0], dtype=int)
    bins[-1] += 1e-6
    return np.digitize(age, bins[1:-1])


def stratified_subsample(
    Z_raw: np.ndarray,
    n_bins: int,
    n_sample: int,
    rng: np.random.Generator,
    age_col: int = 0,
) -> np.ndarray:
    """Return subject indices from stratified sampling by age bins.

    Sampling is without replacement when possible. If `n_sample` exceeds the
    number of available subjects, sampling switches to stratified sampling with
    replacement so the requested sample size is still produced.
    """
    n_total = Z_raw.shape[0]
    if n_sample <= 0:
        raise ValueError(f"n_sample must be positive, got {n_sample}")

    labels = age_strata(Z_raw, n_bins, age_col=age_col)
    unique_bins = np.unique(labels)
    K = len(unique_bins)
    base_per_bin = n_sample // K
    remainder = n_sample - base_per_bin * K

    selected = []
    for rank, b in enumerate(unique_bins):
        idx_b = np.where(labels == b)[0]
        k = base_per_bin + (1 if rank < remainder else 0)
        if k > 0:
            replace = k > len(idx_b)
            selected.append(rng.choice(idx_b, size=k, replace=replace))

    selected = np.concatenate(selected) if selected else np.array([], dtype=int)

    # If some bins were too small in the without-replacement case, top up from
    # the remaining pool. If n_sample > n_total, top up with replacement.
    if selected.size < n_sample:
        need = n_sample - selected.size
        if n_sample > n_total:
            extra = rng.choice(np.arange(n_total), size=need, replace=True)
        else:
            selected_set = set(selected.tolist())
            remaining = np.array([i for i in range(n_total) if i not in selected_set])
            extra = rng.choice(remaining, size=need, replace=False)
        selected = np.concatenate([selected, extra])

    return selected


def permute_age_within_strata(
    Z_raw: np.ndarray,
    n_bins: int,
    rng: np.random.Generator,
    age_col: int = 0,
) -> np.ndarray:
    """Shuffle age column within original age strata; other covariates stay fixed."""
    Z_perm = Z_raw.copy()
    labels = age_strata(Z_raw, n_bins, age_col=age_col)
    age_orig = Z_raw[:, age_col].copy()
    for b in np.unique(labels):
        idx = np.where(labels == b)[0]
        if len(idx) > 1:
            Z_perm[idx, age_col] = age_orig[rng.permutation(idx)]
    return Z_perm


def permute_simple(Z_raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Global row permutation of Z_raw."""
    return Z_raw[rng.permutation(Z_raw.shape[0])]


# ===========================================================================
# Model wrappers: S-GLM
# ===========================================================================

def fit_sglm(
    data_dict: dict,
    marginal_dist: str,
    link_func: str,
    device: torch.device,
    simulated_dset: bool = True,
    alpha: float = 0.05,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> dict:
    """Fit S-GLM and return params dict."""
    if alpha <= 0:
        raise ValueError(f"S-GLM alpha must be positive, got {alpha}")
    if max_iter <= 0:
        raise ValueError(f"S-GLM max_iter must be positive, got {max_iter}")
    BR = BrainRegression_Approximate(
        simulated_dset=simulated_dset,
        dtype=torch.float64,
        device=device,
    )
    BR.load_data(data_dict, "SpatialBrainLesion")
    beta = BR.run_regression(
        model="SpatialBrainLesion",
        marginal_dist=marginal_dist,
        link_func=link_func,
        tol=tol,
        max_iter=max_iter,
        alpha=alpha,
        gradient_mode="dask",
        preconditioner_mode="approximate",
        block_size=5000,
        compute_nll=False,
    )
    return {"beta": beta}


def sglm_pvalues(
    data_dict: dict,
    params: dict,
    inference_method: str,
    polynomial_order: int,
    device: torch.device,
    simulated_dset: bool = True,
    age_col_preprocessed: int = 0,
    marginal_dist: str = "Poisson",
    link_func: str = "log",
    sandwich_meat: str = "null_cluster",
):
    """Run S-GLM inference and return flattened p-values and z-statistics."""
    BI = BrainInference_Approximate(
        model="SpatialBrainLesion",
        marginal_dist=marginal_dist,
        link_func=link_func,
        regression_terms=["multiplicative", "additive"],
        dtype=torch.float64,
        device=device,
    )
    BI.load_params(data=data_dict, params=params)

    if age_col_preprocessed == 0:
        # Existing inference class default contrast appears to test the first
        # covariate column. Keep this path for compatibility and speed.
        BI.create_contrast(
            contrast_vector=None,
            contrast_name="age",
            polynomial_order=polynomial_order,
        )
    else:
        # Assumes BI._R matches the design column order after preprocessing and
        # any intercept addition. Verify this against inference.py for UKB.
        c = np.zeros(BI._R)
        c[age_col_preprocessed] = 1.0
        BI.create_contrast(
            contrast_vector=c,
            contrast_name="age",
            polynomial_order=polynomial_order,
        )

    p_vals, z_stats = BI._glh_con_group(inference_method, sandwich_meat=sandwich_meat)
    return np.asarray(p_vals).ravel(), np.asarray(z_stats).ravel()


# ===========================================================================
# Model wrappers: MUM vectorised Poisson GLM
# ===========================================================================

def _mum_voxel_mask(Y: np.ndarray, min_count: int, min_nonzero: int) -> np.ndarray:
    """Boolean mask of voxels with sufficient support for IRLS."""
    total = Y.sum(axis=0)
    n_nonzero = np.sum(Y > 0, axis=0)
    return (total >= min_count) & (n_nonzero >= min_nonzero)


def mum_pvalues(
    Z_preprocessed: np.ndarray,
    Y: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
    age_col: int = 0,
    min_count: int = 1,
    min_nonzero: int = 1,
):
    """Voxelwise Poisson log-link GLM via vectorised batch IRLS."""
    Z = Z_preprocessed.astype(np.float64)
    Y_full = Y.astype(np.float64)
    n, n_cov = Z.shape
    V_total = Y_full.shape[1]

    if Y_full.shape[0] != n:
        raise ValueError(f"Z and Y row mismatch: Z has {n}, Y has {Y_full.shape[0]}")
    if not (0 <= age_col < n_cov):
        raise ValueError(f"age_col={age_col} out of range for n_cov={n_cov}")

    mask = _mum_voxel_mask(Y_full, min_count=min_count, min_nonzero=min_nonzero)
    n_fit = int(mask.sum())
    n_skipped = V_total - n_fit

    p_out = np.full(V_total, np.nan, dtype=np.float64)
    z_out = np.full(V_total, np.nan, dtype=np.float64)

    if n_fit == 0:
        return p_out, z_out, {"n_fit": 0, "n_skipped": n_skipped, "mask": mask}

    Y_fit = Y_full[:, mask]
    V = n_fit

    beta = np.zeros((n_cov, V), dtype=np.float64)
    beta[-1] = np.log(np.maximum(Y_fit.mean(axis=0), 1e-12))

    ridge_base = 1e-10

    for _ in range(max_iter):
        eta = Z @ beta
        mu = np.exp(np.clip(eta, -30, 30))
        resid = Y_fit - mu

        ZtWZ = np.einsum("ir,ik,iv->rkv", Z, Z, mu)
        score = Z.T @ resid

        if n_cov == 2:
            a = ZtWZ[0, 0]
            b = ZtWZ[0, 1]
            d = ZtWZ[1, 1]

            # Correct ridge for [[a, b], [b, d]]: add reg to diagonal, then invert.
            reg = ridge_base * ((a + d) / 2.0) + 1e-15
            aa = a + reg
            dd = d + reg
            det = aa * dd - b * b

            bad = ~(det > 0) | ~np.isfinite(det)
            det = np.where(bad, np.nan, det)

            delta = np.empty_like(score)
            delta[0] = (dd * score[0] - b * score[1]) / det
            delta[1] = (-b * score[0] + aa * score[1]) / det
        else:
            lhs = ZtWZ.transpose(2, 0, 1)
            tr = np.einsum("vii->v", lhs) / n_cov
            reg = ridge_base * tr[:, None, None] * np.eye(n_cov)[None]
            lhs = lhs + reg
            rhs = score.T[:, :, None]
            delta = np.linalg.solve(lhs, rhs).squeeze(-1).T

        if not np.all(np.isfinite(delta)):
            # Keep finite voxels; non-finite columns will become NaN downstream.
            delta = np.where(np.isfinite(delta), delta, np.nan)

        beta += delta

        max_delta = np.nanmax(np.abs(delta))
        if max_delta < tol:
            break

    eta = Z @ beta
    mu = np.exp(np.clip(eta, -30, 30))
    ZtWZ = np.einsum("ir,ik,iv->rkv", Z, Z, mu)

    lhs = ZtWZ.transpose(2, 0, 1)
    tr = np.einsum("vii->v", lhs) / n_cov
    reg = ridge_base * tr[:, None, None] * np.eye(n_cov)[None]
    lhs = lhs + reg

    eye_k = np.zeros(n_cov)
    eye_k[age_col] = 1.0
    rhs = np.tile(eye_k, (V, 1))[:, :, None]
    col_k = np.linalg.solve(lhs, rhs).squeeze(-1)
    var_age = col_k[:, age_col]

    bad_var = ~(var_age > 0) | ~np.isfinite(var_age)
    se_age = np.sqrt(np.where(bad_var, np.nan, var_age))
    z_fit = beta[age_col] / se_age
    p_fit = 2.0 * scipy.stats.norm.sf(np.abs(z_fit))

    p_out[mask] = p_fit
    z_out[mask] = z_fit
    return (
        p_out.ravel(),
        z_out.ravel(),
        {"n_fit": n_fit, "n_skipped": n_skipped, "mask": mask},
    )


# ===========================================================================
# Reference prediction map and accuracy
# ===========================================================================

def compute_reference_map(
    Z_preprocessed: np.ndarray,
    Y: np.ndarray,
    X_spatial: np.ndarray,
    data_dict: dict,
    marginal_dist: str,
    link_func: str,
    device: torch.device,
    simulated_dset: bool = True,
) -> np.ndarray:
    """Fit S-GLM on full data and return predicted mean map."""
    params = fit_sglm(
        data_dict,
        marginal_dist,
        link_func,
        device,
        simulated_dset=simulated_dset,
    )
    BR = BrainRegression_Approximate.__new__(BrainRegression_Approximate)
    BR.simulated_dset = simulated_dset
    BR.dtype = torch.float64
    BR.device = device
    BR.load_data(data_dict, "SpatialBrainLesion")
    _, _, P_mean = BR.goodness_of_fit(
        beta=params["beta"],
        model="SpatialBrainLesion",
        mode="dask",
        block_size=5000,
    )
    gc.collect()
    return P_mean.ravel()


def compute_accuracy(P_sub: np.ndarray, P_ref: np.ndarray):
    """Return RMSE and Pearson correlation between prediction maps."""
    diff = P_sub - P_ref
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    if np.nanstd(P_sub) == 0 or np.nanstd(P_ref) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(P_sub, P_ref)[0, 1])
    return rmse, corr


# ===========================================================================
# FPR helpers
# ===========================================================================

def compute_fpr(
    p_vals: np.ndarray,
    alpha: float = 0.05,
    apply_fdr: bool = False,
) -> float:
    """Compute FPR among finite p-values."""
    p = np.asarray(p_vals).ravel()
    valid = np.isfinite(p)
    if not valid.any():
        return float("nan")
    p_v = p[valid]
    if apply_fdr:
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(p_v, alpha=alpha, method="fdr_bh")
        sig = np.sum(p_adj < alpha)
    else:
        sig = int(np.sum(p_v < alpha))
    return float(sig / len(p_v))


# ===========================================================================
# Single repetition worker
# ===========================================================================

def run_one_rep(
    rep: int,
    N: int,
    base_data: dict,
    args: argparse.Namespace,
    master_seed: int,
) -> dict:
    """Run one subsampling + permutation repetition."""
    rng = np.random.default_rng(master_seed)
    device = torch.device("cpu")

    simulated_dset = not args.use_ukb
    age_col_raw = base_data.get("age_col", 0)

    Z_raw = base_data["Z"]
    Y_full = base_data["Y"]
    X_spatial = base_data["X_spatial"]

    idx = stratified_subsample(
        Z_raw,
        n_bins=args.n_age_bins,
        n_sample=N,
        rng=rng,
        age_col=age_col_raw,
    )
    Z_sub_raw = Z_raw[idx]
    Y_sub = Y_full[idx]

    Z_sub_pre = preprocess_Z(
        simulated_dset=simulated_dset,
        Z=Z_sub_raw,
        polynomial_order=args.polynomial_order,
    )
    data_sub = make_multigroup_data(Z_sub_pre, Y_sub, X_spatial)

    # Expected preprocessed order:
    #   GRF: [age, age^2, ...]
    #   UKB: [sex, age, headsize, CVR, ...]
    # Verify against util.preprocess_Z if that function changes.
    age_col_preprocessed = 0 if simulated_dset else 1

    result = {"N": N, "rep": rep}

    # -------------------------------------------------------------------
    # Real-data S-GLM
    # -------------------------------------------------------------------
    sglm_params = fit_sglm(
        data_sub,
        args.marginal_dist,
        args.link_func,
        device,
        simulated_dset=simulated_dset,
        alpha=args.sglm_alpha,
        max_iter=args.sglm_max_iter,
        tol=args.sglm_tol,
    )

    try:
        p_sglm_real, z_sglm_real = sglm_pvalues(
            data_sub,
            sglm_params,
            args.inference_method,
            args.polynomial_order,
            device,
            simulated_dset=simulated_dset,
            age_col_preprocessed=age_col_preprocessed,
            marginal_dist=args.marginal_dist,
            link_func=args.link_func,
            sandwich_meat=args.sglm_sandwich_meat,
        )
        result["p_sglm_real"] = p_sglm_real.astype(np.float32)
        result["z_sglm_real"] = z_sglm_real.astype(np.float32)
        del p_sglm_real, z_sglm_real
    except Exception as exc:
        logger.warning("N=%d rep=%d: S-GLM real-data inference failed -- %s", N, rep, exc)
        result["p_sglm_real"] = None

    # -------------------------------------------------------------------
    # Real-data MUM
    # -------------------------------------------------------------------
    Z_sub_mum = np.concatenate([Z_sub_pre, np.ones((Z_sub_pre.shape[0], 1))], axis=1)
    try:
        p_mum_real, z_mum_real, info_real = mum_pvalues(
            Z_sub_mum,
            Y_sub.astype(np.float64),
            age_col=age_col_preprocessed,
            min_count=args.mum_min_count,
            min_nonzero=args.mum_min_nonzero,
        )
        result["p_mum_real"] = p_mum_real.astype(np.float32)
        result["z_mum_real"] = z_mum_real.astype(np.float32)
        result["mum_real_n_fit"] = info_real["n_fit"]
        result["mum_real_n_skip"] = info_real["n_skipped"]
        del p_mum_real, z_mum_real
    except Exception as exc:
        logger.warning("N=%d rep=%d: MUM real-data inference failed -- %s", N, rep, exc)
        result["p_mum_real"] = None

    # -------------------------------------------------------------------
    # S-GLM prediction map for accuracy/posthoc use
    # -------------------------------------------------------------------
    try:
        BR_sub = BrainRegression_Approximate(
            simulated_dset=simulated_dset,
            dtype=torch.float64,
            device=device,
        )
        BR_sub.load_data(data_sub, "SpatialBrainLesion")
        _, _, P_sub = BR_sub.goodness_of_fit(
            beta=sglm_params["beta"],
            model="SpatialBrainLesion",
            mode="dask",
            block_size=5000,
        )
        result["P_sub_sglm"] = P_sub.ravel().astype(np.float32)
        del BR_sub, P_sub
        gc.collect()
    except Exception as exc:
        logger.warning("N=%d rep=%d: goodness_of_fit failed -- %s", N, rep, exc)
        result["P_sub_sglm"] = None

    # -------------------------------------------------------------------
    # Null test: permute covariates / age and rerun inference
    # -------------------------------------------------------------------
    if args.permutation_scheme == "stratified_age":
        Z_perm_raw = permute_age_within_strata(
            Z_sub_raw,
            n_bins=args.n_age_bins,
            rng=rng,
            age_col=age_col_raw,
        )
    else:
        Z_perm_raw = permute_simple(Z_sub_raw, rng)

    Z_perm_pre = preprocess_Z(
        simulated_dset=simulated_dset,
        Z=Z_perm_raw,
        polynomial_order=args.polynomial_order,
    )
    data_perm = make_multigroup_data(Z_perm_pre, Y_sub, X_spatial)

    try:
        sglm_perm_params = fit_sglm(
            data_perm,
            args.marginal_dist,
            args.link_func,
            device,
            simulated_dset=simulated_dset,
            alpha=args.sglm_alpha,
            max_iter=args.sglm_max_iter,
            tol=args.sglm_tol,
        )
        p_sglm, z_sglm = sglm_pvalues(
            data_perm,
            sglm_perm_params,
            args.inference_method,
            args.polynomial_order,
            device,
            simulated_dset=simulated_dset,
            age_col_preprocessed=age_col_preprocessed,
            marginal_dist=args.marginal_dist,
            link_func=args.link_func,
            sandwich_meat=args.sglm_sandwich_meat,
        )
        result["p_sglm"] = p_sglm.astype(np.float32)
        result["z_sglm"] = z_sglm.astype(np.float32)
        result["fpr_sglm"] = compute_fpr(p_sglm, args.alpha_threshold, apply_fdr=False)
        if args.fdr:
            result["fpr_sglm_fdr"] = compute_fpr(p_sglm, args.alpha_threshold, apply_fdr=True)
        del p_sglm, z_sglm, sglm_perm_params
        gc.collect()
    except Exception as exc:
        logger.warning("N=%d rep=%d: S-GLM null inference failed -- %s", N, rep, exc)
        result["p_sglm"] = None
        result["fpr_sglm"] = float("nan")

    try:
        Z_perm_mum = np.concatenate([Z_perm_pre, np.ones((Z_perm_pre.shape[0], 1))], axis=1)
        p_mum, z_mum, info_null = mum_pvalues(
            Z_perm_mum,
            Y_sub.astype(np.float64),
            age_col=age_col_preprocessed,
            min_count=args.mum_min_count,
            min_nonzero=args.mum_min_nonzero,
        )
        result["p_mum"] = p_mum.astype(np.float32)
        result["z_mum"] = z_mum.astype(np.float32)
        result["mum_null_n_fit"] = info_null["n_fit"]
        result["mum_null_n_skip"] = info_null["n_skipped"]
        result["fpr_mum"] = compute_fpr(p_mum, args.alpha_threshold, apply_fdr=False)
        if args.fdr:
            result["fpr_mum_fdr"] = compute_fpr(p_mum, args.alpha_threshold, apply_fdr=True)
        del p_mum, z_mum, Z_perm_mum
        gc.collect()
    except Exception as exc:
        logger.warning("N=%d rep=%d: MUM null inference failed -- %s", N, rep, exc)
        result["p_mum"] = None
        result["fpr_mum"] = float("nan")

    logger.info(
        "N=%4d  rep=%3d  FPR_SGLM=%.4f  FPR_MUM=%.4f  MUM_skip=%s",
        N,
        rep,
        result.get("fpr_sglm", float("nan")),
        result.get("fpr_mum", float("nan")),
        result.get("mum_null_n_skip", "?"),
    )
    return result


# ===========================================================================
# Aggregation and saving
# ===========================================================================

def aggregate_metrics(all_results: list, args: argparse.Namespace) -> List[Dict]:
    """Aggregate raw results into per-(N, method) summary rows."""
    rows = []
    grouped = defaultdict(lambda: defaultdict(list))
    fdr_grouped = defaultdict(lambda: defaultdict(list))

    for r in all_results:
        N = r["N"]
        for method in ("sglm", "mum"):
            fpr_key = f"fpr_{method}"
            if fpr_key in r and np.isfinite(r[fpr_key]):
                grouped[N][method].append(r[fpr_key])
            fdr_key = f"fpr_{method}_fdr"
            if fdr_key in r and np.isfinite(r[fdr_key]):
                fdr_grouped[N][method].append(r[fdr_key])

    for N in sorted(grouped):
        for method, fprs in grouped[N].items():
            row = {
                "N": N,
                "method": method.upper(),
                "fpr_mean": float(np.mean(fprs)),
                "fpr_std": float(np.std(fprs)),
                "fpr_min": float(np.min(fprs)),
                "fpr_max": float(np.max(fprs)),
                "n_reps": len(fprs),
            }
            fdrs = fdr_grouped[N].get(method, [])
            if fdrs:
                row.update({
                    "fpr_fdr_mean": float(np.mean(fdrs)),
                    "fpr_fdr_std": float(np.std(fdrs)),
                    "fpr_fdr_min": float(np.min(fdrs)),
                    "fpr_fdr_max": float(np.max(fdrs)),
                })
            rows.append(row)
    return rows


def save_one_rep(result: dict, ts: str):
    """Save a single repetition's results immediately after it completes."""
    N = result["N"]
    rep = result["rep"]
    seed = result.get("seed")
    if seed is not None:
        out_path = os.path.join(RAW_DIR, f"rep_N{N}_rep{rep}_seed{seed}.npz")
    else:
        out_path = os.path.join(RAW_DIR, f"rep_N{N}_rep{rep}_{ts}.npz")

    payload = {}
    array_keys = (
        "p_sglm",
        "z_sglm",
        "p_mum",
        "z_mum",
        "P_sub_sglm",
        "p_mum_real",
        "z_mum_real",
        "p_sglm_real",
        "z_sglm_real",
    )
    scalar_keys = (
        "fpr_sglm",
        "fpr_mum",
        "fpr_sglm_fdr",
        "fpr_mum_fdr",
        "mum_real_n_fit",
        "mum_real_n_skip",
        "mum_null_n_fit",
        "mum_null_n_skip",
    )

    for key in array_keys:
        val = result.get(key)
        if val is not None:
            payload[key] = val
    for key in scalar_keys:
        if key in result:
            payload[key] = np.array(result[key])

    payload["N"] = np.array(N)
    payload["rep"] = np.array(rep)
    if seed is not None:
        payload["seed"] = np.array(seed)

    np.savez_compressed(out_path, **payload)
    logger.info("Saved rep result -> %s", out_path)


def save_raw_results(all_results: list, N: int, ts: str):
    """Save per-rep raw arrays for a given N in one aggregated file."""
    out_path = os.path.join(RAW_DIR, f"raw_N{N}_{ts}.npz")
    payload = {}

    array_keys = (
        "p_sglm",
        "z_sglm",
        "p_mum",
        "z_mum",
        "P_sub_sglm",
        "p_mum_real",
        "z_mum_real",
        "p_sglm_real",
        "z_sglm_real",
    )
    scalar_keys = (
        "fpr_sglm",
        "fpr_mum",
        "fpr_sglm_fdr",
        "fpr_mum_fdr",
        "mum_real_n_fit",
        "mum_real_n_skip",
        "mum_null_n_fit",
        "mum_null_n_skip",
    )

    for r in all_results:
        if r["N"] != N:
            continue
        rep = r["rep"]
        for key in array_keys:
            val = r.get(key)
            if val is not None:
                payload[f"{key}_rep{rep}"] = val
        for key in scalar_keys:
            if key in r:
                payload[f"{key}_rep{rep}"] = np.array(r[key])

    np.savez_compressed(out_path, **payload)
    logger.info("Saved raw results -> %s", out_path)


def save_metrics_csv(rows: list, ts: str):
    """Write aggregated metrics to CSV."""
    if not rows:
        return None

    csv_path = os.path.join(METRICS_DIR, f"metrics_{ts}.csv")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "N",
        "method",
        "fpr_mean",
        "fpr_std",
        "fpr_min",
        "fpr_max",
        "fpr_fdr_mean",
        "fpr_fdr_std",
        "fpr_fdr_min",
        "fpr_fdr_max",
        "n_reps",
    ]
    fieldnames = [f for f in preferred if f in fieldnames] + [
        f for f in fieldnames if f not in preferred
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved metrics CSV -> %s", csv_path)
    return csv_path


# ===========================================================================
# PP-plot generation
# ===========================================================================

def _pp_curve_linear(ax, p_list, colour, label):
    if not p_list:
        return
    p_cat = np.concatenate(p_list)
    p_cat = p_cat[np.isfinite(p_cat)]
    if len(p_cat) == 0:
        return
    p_sorted = np.sort(p_cat)
    n_pts = len(p_sorted)
    expected = np.linspace(0, 1, n_pts)
    ax.plot(
        expected,
        p_sorted,
        color=colour,
        lw=1.2,
        label=f"{label}  (n={n_pts:,})",
        alpha=0.85,
    )


def _pp_curve_log(ax, p_list, colour, label):
    if not p_list:
        return
    p_cat = np.concatenate(p_list)
    p_cat = p_cat[np.isfinite(p_cat) & (p_cat > 0)]
    if len(p_cat) == 0:
        return
    p_sorted = np.sort(p_cat)
    n_pts = len(p_sorted)
    expected = np.linspace(1.0 / (n_pts + 1), n_pts / (n_pts + 1), n_pts)
    ax.plot(
        -np.log10(expected),
        -np.log10(p_sorted),
        color=colour,
        lw=1.2,
        label=f"{label}  (n={n_pts:,})",
        alpha=0.85,
    )


def make_pp_plot(all_results: list, N: int, args: argparse.Namespace, ts: str):
    """Generate and save linear and -log10 PP-plots for a given N."""
    p_sglm_null, p_mum_null = [], []
    p_sglm_real, p_mum_real = [], []

    for r in all_results:
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

    if not (p_sglm_null or p_mum_null or p_sglm_real or p_mum_real):
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    panels = [
        (p_sglm_null, p_mum_null, f"PP-plot -- AGE null, permuted | N={N} R={args.R}"),
        (p_sglm_real, p_mum_real, f"PP-plot -- AGE real data | N={N} R={args.R}"),
    ]
    for ax, (psg, pmu, title) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal uniform")
        _pp_curve_linear(ax, psg, "#E64646", "S-GLM")
        _pp_curve_linear(ax, pmu, "#4682B4", "MUM")
        ax.set_xlabel("Expected quantile, Uniform")
        ax.set_ylabel("Observed p-value quantile")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    out_lin = os.path.join(PLOTS_DIR, f"PPplot_N{N}_{ts}_linear.png")
    fig.savefig(out_lin, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (psg, pmu, title) in zip(axes, panels):
        all_p = []
        for plist in (psg, pmu):
            if plist:
                pc = np.concatenate(plist)
                pc = pc[np.isfinite(pc) & (pc > 0)]
                if len(pc):
                    all_p.append(pc.min())
        lim = -np.log10(min(all_p)) if all_p else 5.0
        lim = max(lim * 1.05, 1.0)
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="Ideal uniform")
        _pp_curve_log(ax, psg, "#E64646", "S-GLM")
        _pp_curve_log(ax, pmu, "#4682B4", "MUM")
        ax.set_xlabel("Expected quantile, -log10 p")
        ax.set_ylabel("Observed quantile, -log10 p")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    fig.tight_layout()
    out_log = os.path.join(PLOTS_DIR, f"PPplot_N{N}_{ts}_log.png")
    fig.savefig(out_log, dpi=150)
    plt.close(fig)

    logger.info("Saved PP-plots -> %s , %s", out_lin, out_log)


def make_fpr_summary_plot(rows: list, alpha: float, ts: str):
    """Bar chart of mean FPR +/- std for S-GLM vs MUM across N."""
    if not rows:
        return
    N_vals = sorted(set(r["N"] for r in rows))
    methods = ["SGLM", "MUM"]
    colours = {"SGLM": "#E64646", "MUM": "#4682B4"}
    x = np.arange(len(N_vals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, method in enumerate(methods):
        means = []
        stds = []
        for N in N_vals:
            row = next((r for r in rows if r["N"] == N and r["method"] == method), None)
            means.append(row["fpr_mean"] if row else float("nan"))
            stds.append(row["fpr_std"] if row else 0.0)
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=method,
            color=colours[method],
            alpha=0.8,
        )

    ax.axhline(alpha, color="k", linestyle="--", lw=1, label=f"alpha={alpha}")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(n) for n in N_vals])
    ax.set_xlabel("Subsample size N")
    ax.set_ylabel("Mean FPR")
    ax.set_title("FPR under null, age permuted: S-GLM vs MUM")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"FPR_summary_{ts}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved FPR summary plot -> %s", out_path)


# ===========================================================================
# Main
# ===========================================================================

def _make_seed(base_seed: int, N: int, rep: int) -> int:
    """Generate a robust per-rep seed for each (base_seed, N, rep)."""
    ss = np.random.SeedSequence([int(base_seed), int(N), int(rep)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _resolve_N_list(N_list: List[int], n_full: int) -> List[int]:
    """Validate N values and remove duplicates to avoid repeated overwrites."""
    resolved = []
    seen = set()
    for N in N_list:
        if N <= 0:
            raise ValueError(f"All N values must be positive; got {N}")
        if N in seen:
            logger.warning("Skipping duplicate N=%d to avoid overwriting outputs", N)
            continue
        if N > n_full:
            logger.info(
                "N=%d > n_full=%d -- stratified sampling with replacement will be used",
                N,
                n_full,
            )
        resolved.append(N)
        seen.add(N)
    return resolved


def main():
    args = get_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    single_seed_mode = args.seed is not None
    single_rep_mode = (args.rep is not None) and not single_seed_mode

    _setup_output_dirs(args.R, use_ukb=args.use_ukb)

    if args.rep is not None and args.rep < 0:
        raise ValueError(f"--rep must be >= 0, got {args.rep}")

    if single_rep_mode and (args.rep_start != 0 or args.rep_count is not None):
        logger.warning(
            "--rep was provided, so --rep_start/--rep_count are ignored. Running only rep=%d.",
            args.rep,
        )

    if not single_seed_mode and not single_rep_mode:
        rep_count = args.rep_count if args.rep_count is not None else args.R
        if args.rep_start < 0:
            raise ValueError(f"--rep_start must be >= 0, got {args.rep_start}")
        if rep_count <= 0:
            raise ValueError(f"--rep_count must be >= 1, got {rep_count}")
        if args.rep_start + rep_count > args.R:
            logger.warning(
                "--rep_start (%d) + --rep_count (%d) = %d exceeds --R (%d). "
                "This is allowed, but rep indices >= R will be used. Make sure --R "
                "reflects the intended global total across all devices.",
                args.rep_start,
                rep_count,
                args.rep_start + rep_count,
                args.R,
            )

    file_handler = logging.FileHandler(os.path.join(LOGS_DIR, f"experiment_{ts}.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logging.getLogger().addHandler(file_handler)

    config = vars(args).copy()
    config["timestamp"] = ts
    config["experiment_dir"] = EXPERIMENT_OUT
    cfg_path = os.path.join(LOGS_DIR, f"config_{ts}.json")
    with open(cfg_path, "w") as fh:
        json.dump(config, fh, indent=2)
    logger.info("Experiment config: %s", json.dumps(config, indent=2))
    if args.use_ukb:
        logger.info("Loading UKB real dataset, spacing=%d", args.spacing)
        base_data = load_ukb_data(spacing=args.spacing)
    else:
        logger.info(
            "Loading GRF base data, n_subject=%d, seed=%d",
            args.data_n_subject,
            args.data_seed,
        )
        base_data = load_base_data(args.data_n_subject, args.data_seed)

    n_full = base_data["Z"].shape[0]
    logger.info(
        "%s data loaded: n=%d, V=%d, P=%d",
        "UKB" if args.use_ukb else "GRF",
        n_full,
        base_data["Y"].shape[1],
        base_data["X_spatial"].shape[1],
    )

    resolved_N_list = _resolve_N_list(args.N_list, n_full)
    logger.info("Effective N_list: %s", resolved_N_list)

    all_results = []

    for N in resolved_N_list:
        if single_seed_mode:
            rep_label = args.rep if args.rep is not None else 0
            rep_indices = [rep_label]
            rep_start = rep_label
            rep_end = rep_label + 1
            rep_count = 1
        elif single_rep_mode:
            rep_indices = [args.rep]
            rep_start = args.rep
            rep_end = args.rep + 1
            rep_count = 1
        else:
            rep_start = args.rep_start
            rep_count = args.rep_count if args.rep_count is not None else args.R
            rep_end = rep_start + rep_count
            rep_indices = list(range(rep_start, rep_end))

        logger.info("=" * 60)
        if single_seed_mode:
            logger.info("Subsample size N=%d, single-seed run, rep=%d", N, rep_indices[0])
        elif single_rep_mode:
            logger.info("Subsample size N=%d, single deterministic rep=%d", N, rep_indices[0])
        elif rep_start == 0 and rep_count == args.R:
            logger.info("Subsample size N=%d, R=%d repetitions", N, args.R)
        else:
            logger.info(
                "Subsample size N=%d, running reps %d-%d of global R=%d",
                N,
                rep_start,
                rep_end - 1,
                args.R,
            )
        logger.info("=" * 60)

        t0 = time.time()
        rep_results = []

        if single_seed_mode:
            res = run_one_rep(
                rep=rep_indices[0],
                N=N,
                base_data=base_data,
                args=args,
                master_seed=args.seed,
            )
            res["seed"] = args.seed
            save_one_rep(res, ts)
            rep_results.append(res)
        elif single_rep_mode:
            seed = _make_seed(args.base_seed, N, args.rep)
            res = run_one_rep(
                rep=args.rep,
                N=N,
                base_data=base_data,
                args=args,
                master_seed=seed,
            )
            res["seed"] = seed
            save_one_rep(res, ts)
            rep_results.append(res)
        else:
            seeds = {r: _make_seed(args.base_seed, N, r) for r in rep_indices}
            n_jobs = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
            n_jobs = max(1, int(n_jobs))

            if args.use_ukb and n_jobs > 1:
                logger.warning(
                    "UKB with n_jobs=%d may duplicate large arrays across worker processes. "
                    "Use n_jobs=1 if memory is limited.",
                    n_jobs,
                )

            worker = functools.partial(run_one_rep, N=N, base_data=base_data, args=args)

            if n_jobs == 1:
                for r in rep_indices:
                    res = worker(rep=r, master_seed=seeds[r])
                    res["seed"] = seeds[r]
                    save_one_rep(res, ts)
                    rep_results.append(res)
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = {
                        executor.submit(worker, rep=r, master_seed=seeds[r]): r
                        for r in rep_indices
                    }
                    for fut in concurrent.futures.as_completed(futures):
                        r = futures[fut]
                        try:
                            res = fut.result()
                            res["seed"] = seeds[r]
                            save_one_rep(res, ts)
                            rep_results.append(res)
                        except Exception as exc:
                            logger.error("N=%d rep=%d raised an exception: %s", N, r, exc)

        all_results.extend(rep_results)
        if args.save_aggregated_npz:
            save_raw_results(rep_results, N, ts)

        elapsed = time.time() - t0
        fpr_sglm_vals = [
            r["fpr_sglm"] for r in rep_results
            if "fpr_sglm" in r and np.isfinite(r["fpr_sglm"])
        ]
        fpr_mum_vals = [
            r["fpr_mum"] for r in rep_results
            if "fpr_mum" in r and np.isfinite(r["fpr_mum"])
        ]
        logger.info(
            "N=%d done in %.1fs | FPR_SGLM=%.4f+/-%.4f FPR_MUM=%.4f+/-%.4f",
            N,
            elapsed,
            np.mean(fpr_sglm_vals) if fpr_sglm_vals else float("nan"),
            np.std(fpr_sglm_vals) if fpr_sglm_vals else 0.0,
            np.mean(fpr_mum_vals) if fpr_mum_vals else float("nan"),
            np.std(fpr_mum_vals) if fpr_mum_vals else 0.0,
        )
        del rep_results
        gc.collect()

    rows = aggregate_metrics(all_results, args)
    save_metrics_csv(rows, ts)

    logger.info("\n%s", "-" * 62)
    logger.info("%-8s  %-8s  %10s  %10s  %6s", "N", "Method", "FPR mean", "FPR std", "n_reps")
    logger.info("%s", "-" * 62)
    for row in rows:
        logger.info(
            "%-8d  %-8s  %10.4f  %10.4f  %6d",
            row["N"],
            row["method"],
            row["fpr_mean"],
            row["fpr_std"],
            row["n_reps"],
        )
    logger.info("%s", "-" * 62)
    logger.info("Experiment complete. Outputs in: %s", EXPERIMENT_OUT)


if __name__ == "__main__":
    main()
