"""
Subsampling + Permutation FPR Analysis: S-GLM vs MUM
=====================================================
Evaluates false positive rate (FPR) for the AGE covariate under the null
hypothesis (permuted age) across different subsample sizes.

Usage
-----
python run_subsampling_experiment.py \
    --N_list 50 100 200 \
    --R 50 \
    --inference_method sandwich \
    --base_seed 42

Does NOT modify any existing code or result folders.
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
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.stats
import torch

# ---------------------------------------------------------------------------
# Add experiment directory to path so existing modules are importable
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))   # …/experiments/
PARENT_DIR     = os.path.dirname(EXPERIMENT_DIR)               # …/experiment/
for _p in (PARENT_DIR, EXPERIMENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from regression import BrainRegression_Approximate
from inference import BrainInference_Approximate
from util import preprocess_Z

# ---------------------------------------------------------------------------
# Paths (mirrors run.py conventions, never writes here)
# ---------------------------------------------------------------------------
GRF_DATA_DIR = os.path.join(PARENT_DIR, "data", "brain", "data_Simulation")
MASK_PATH = os.path.join(PARENT_DIR, "data", "brain",
                         "smooth_lesion_mask_Simulation.nii.gz")

# UKB real dataset paths
UKB_DATA_DIR     = os.path.join(PARENT_DIR, "data", "UKB")
UKB_MASKED_DATA  = os.path.join(UKB_DATA_DIR, "masked_data_RealDataset_spacing_{spacing}.npz")

# Output directories — resolved in main() once args (including R) are known.
EXPERIMENT_OUT = None
RAW_DIR        = None
METRICS_DIR    = None
PLOTS_DIR      = None
LOGS_DIR       = None

# ---------------------------------------------------------------------------
# Logging (console only until output dirs are known; file handler added in main)
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
        EXPERIMENT_DIR, f"subsampling_experiment_{dataset_tag}_R{R}"
    )
    RAW_DIR     = os.path.join(EXPERIMENT_OUT, "results", "raw")
    METRICS_DIR = os.path.join(EXPERIMENT_OUT, "results", "metrics")
    PLOTS_DIR   = os.path.join(EXPERIMENT_OUT, "results", "plots")
    LOGS_DIR    = os.path.join(EXPERIMENT_OUT, "logs")
    for _d in [RAW_DIR, METRICS_DIR, PLOTS_DIR, LOGS_DIR]:
        os.makedirs(_d, exist_ok=True)


# ===========================================================================
# Argument parsing
# ===========================================================================

def get_args():
    p = argparse.ArgumentParser(description="Subsampling FPR experiment")
    p.add_argument("--N_list", nargs="+", type=int, default=[50, 100, 200, 500, 1000, 2000],
                   help="Subsample sizes to evaluate")
    p.add_argument("--R", type=int, default=50,
                   help="Number of repetitions per subsample size (ignored when --seed is set)")
    p.add_argument("--base_seed", type=int, default=42,
                   help="Master random seed (used when --seed is not set)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for a single realization. When provided the script "
                        "runs exactly ONE realization per N and exits. "
                        "Use --rep to label the repetition index for output file naming.")
    p.add_argument("--rep", type=int, default=0,
                   help="Repetition index used for output file naming when --seed is set.")
    p.add_argument("--use_ukb", action="store_true",
                   help="Use the UKB real dataset instead of GRF simulated data")
    p.add_argument("--data_n_subject", type=int, default=100,
                   help="n_subject tag of the GRF data files  (e.g. 100 → GRF_[100]); ignored when --use_ukb")
    p.add_argument("--data_seed", type=int, default=0,
                   help="Random seed of the source GRF data file to use as base; ignored when --use_ukb")
    p.add_argument("--spacing", type=int, default=5,
                   help="B-spline spacing (must match how X_spatial was built)")
    p.add_argument("--polynomial_order", type=int, default=1,
                   help="Polynomial order for preprocess_Z (1 = linear age)")
    p.add_argument("--inference_method", type=str, default="sandwich",
                   choices=["FI", "sandwich"],
                   help="Variance estimator for S-GLM inference")
    p.add_argument("--marginal_dist", type=str, default="Poisson")
    p.add_argument("--link_func", type=str, default="log")
    p.add_argument("--alpha_threshold", type=float, default=0.05,
                   help="Significance threshold for FPR computation")
    p.add_argument("--n_age_bins", type=int, default=5,
                   help="Number of age bins for stratification")
    p.add_argument("--fdr", action="store_true",
                   help="Also compute FPR after BH-FDR correction")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Number of parallel worker processes (-1 = use all CPUs)")
    return p.parse_args()


# ===========================================================================
# Data utilities
# ===========================================================================

def load_ukb_data(spacing: int = 5) -> dict:
    """Load the UKB real dataset (masked NPZ) and return raw numpy arrays.

    The NPZ contains:
      Z          : (N, 5)  columns = [subjectID, sex, age, headsize, CVR]
      Y          : (N, V)  lesion counts per voxel
      X_spatial  : (V, P)  B-spline spatial basis

    Returns
    -------
    dict with keys:
      "Z"         (N, 5) raw covariate matrix (first col is subject ID)
      "Y"         (N, V) count observations
      "X_spatial" (V, P) spatial basis
      "age_col"   int    index of the age column in Z (= 2)
    """
    fname = UKB_MASKED_DATA.format(spacing=spacing)
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"UKB masked data file not found: {fname}\n"
            f"Expected spacing={spacing}.  Available files in {UKB_DATA_DIR}:\n"
            + "\n".join(os.listdir(UKB_DATA_DIR))
        )
    raw = np.load(fname, allow_pickle=True)
    return {
        "Z":         raw["Z"].astype(np.float64),         # (N, 5)
        "Y":         raw["Y"].astype(np.float32),          # (N, V)
        "X_spatial": raw["X_spatial"].astype(np.float64), # (V, P)
        "age_col":   2,   # age is column 2 in the UKB Z matrix
    }


def load_base_data(n_subject: int, data_seed: int) -> dict:
    """Load a GRF simulation NPZ file and return raw numpy arrays.

    Returns
    -------
    dict with keys: "Z" (n, 1) raw ages, "Y" (n, N_voxels), "X_spatial" (N_voxels, P)
    """
    folder = os.path.join(GRF_DATA_DIR, f"GRF_[{n_subject}]")
    fname  = os.path.join(folder, f"GRF_[{n_subject}]_random_seed_{data_seed}.npz")
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Data file not found: {fname}\n"
            f"Run data generation first (run.py --run_data_generation True)."
        )
    raw = np.load(fname, allow_pickle=True)
    # Multi-group format: first key wraps a 0-D object array -> dict
    group_key = list(raw.files)[0]
    group_dict = raw[group_key].item()
    return {
        "Z": group_dict["Z"].astype(np.float64),          # (n, 1) raw ages
        "Y": group_dict["Y"].astype(np.float32),           # (n, N_voxels)
        "X_spatial": group_dict["X_spatial"].astype(np.float64),  # (N_voxels, P)
    }


def make_multigroup_data(Z: np.ndarray, Y: np.ndarray,
                         X_spatial: np.ndarray,
                         group_name: str = "Group_1") -> dict:
    """Pack arrays into the multi-group NPZ dict format expected by existing classes."""
    inner = {"Z": Z, "Y": Y, "X_spatial": X_spatial}
    return {group_name: np.array(inner, dtype=object)}


# ===========================================================================
# Stratification utilities
# ===========================================================================

def age_strata(Z_raw: np.ndarray, n_bins: int, age_col: int = 0) -> np.ndarray:
    """Return integer stratum labels (0 … n_bins-1) for each subject.

    Parameters
    ----------
    Z_raw   : raw covariate matrix
    n_bins  : number of quantile-based bins
    age_col : column index of the age covariate in Z_raw
                (0 for GRF simulated data, 2 for UKB real data)
    """
    age = Z_raw[:, age_col]
    bins = np.quantile(age, np.linspace(0.0, 1.0, n_bins + 1))
    bins[-1] += 1e-6  # ensure the maximum age falls in the last bin
    return np.digitize(age, bins[1:-1])  # 0-indexed labels


def stratified_subsample(Z_raw: np.ndarray, n_bins: int,
                         n_sample: int, rng: np.random.Generator,
                         age_col: int = 0) -> np.ndarray:
    """Return sorted subject indices from stratified sampling by age bins.

    Sampling is proportional to bin size (round-robin remainder allocation).

    Parameters
    ----------
    age_col : column index of the age covariate in Z_raw
                (0 for GRF simulated data, 2 for UKB real data)
    """
    n_total = Z_raw.shape[0]
    if n_sample >= n_total:
        return np.arange(n_total)

    labels = age_strata(Z_raw, n_bins, age_col=age_col)
    unique_bins = np.unique(labels)
    K = len(unique_bins)
    base_per_bin = n_sample // K
    remainder    = n_sample - base_per_bin * K

    selected = []
    for rank, b in enumerate(unique_bins):
        idx_b = np.where(labels == b)[0]
        k = base_per_bin + (1 if rank < remainder else 0)
        k = min(k, len(idx_b))
        selected.append(rng.choice(idx_b, size=k, replace=False))

    return np.sort(np.concatenate(selected))


def permute_age(Z_preprocessed: np.ndarray, Z_raw: np.ndarray,
                n_bins: int, rng: np.random.Generator,
                age_col: int = 0) -> np.ndarray:
    """Shuffle the age column within age strata (permutation within strata).

    Parameters
    ----------
    Z_preprocessed : (n, R) standardised Z (from preprocess_Z) — will be permuted
    Z_raw          : (n, C) raw covariate matrix used only for computing strata labels
    n_bins         : number of age bins used as strata
    rng            : numpy random generator
    age_col        : column index of age in Z_raw (0 for GRF, 2 for UKB)

    Returns
    -------
    Z_perm : (n, R) copy of Z_preprocessed with age column shuffled within strata
    """
    Z_perm   = Z_preprocessed.copy()
    labels   = age_strata(Z_raw, n_bins, age_col=age_col)
    for b in np.unique(labels):
        idx = np.where(labels == b)[0]
        shuffled = rng.permutation(idx)
        Z_perm[idx, 0] = Z_preprocessed[shuffled, 0]
    return Z_perm


# ===========================================================================
# Model wrappers — S-GLM (BrainRegression_Approximate + BrainInference_Approximate)
# ===========================================================================

def fit_sglm(data_dict: dict, marginal_dist: str, link_func: str,
             device: torch.device, simulated_dset: bool = True) -> dict:
    """Fit S-GLM and return params dict {"beta", ...}."""
    BR = BrainRegression_Approximate(simulated_dset=simulated_dset,
                                     dtype=torch.float64, device=device)
    BR.load_data(data_dict, "SpatialBrainLesion")
    beta = BR.run_regression(
        model="SpatialBrainLesion",
        marginal_dist=marginal_dist,
        link_func=link_func,
        max_iter=500,
        alpha=0.01,
        gradient_mode="dask",
        preconditioner_mode="approximate",
        block_size=5000,
        compute_nll=False,
    )
    return {"beta": beta}


def sglm_pvalues(data_dict: dict, params: dict,
                 inference_method: str, polynomial_order: int,
                 device: torch.device, simulated_dset: bool = True,
                 age_col_preprocessed: int = 0):
    """Run S-GLM inference and return (p_vals, z_stats) as 1-D arrays.

    Parameters
    ----------
    age_col_preprocessed : int
        Column index of the age covariate in the *preprocessed* Z matrix.
        For GRF (simulated) data this is 0 (age is the first column).
        For UKB real data this is 1 (sex is column 0, age is column 1).
    """
    BI = BrainInference_Approximate(
        model="SpatialBrainLesion",
        marginal_dist="Poisson",
        link_func="log",
        regression_terms=["multiplicative", "additive"],
        dtype=torch.float64,
        device=device,
    )
    BI.load_params(data=data_dict, params=params)
    # Build contrast vector: tests only the age column.
    # BI._R is the number of covariates (including intercept added by load_params).
    if age_col_preprocessed == 0:
        # Default: test first column — pass contrast_vector=None for efficiency.
        BI.create_contrast(contrast_vector=None, contrast_name="age",
                           polynomial_order=polynomial_order)
    else:
        c = np.zeros(BI._R)
        c[age_col_preprocessed] = 1.0
        BI.create_contrast(contrast_vector=c, contrast_name="age",
                           polynomial_order=polynomial_order)
    p_vals, z_stats = BI._glh_con_group(inference_method)
    # p_vals / z_stats may have shape (1, N_voxels) → flatten to (N_voxels,)
    return np.asarray(p_vals).ravel(), np.asarray(z_stats).ravel()


# ===========================================================================
# Model wrappers — MUM (vectorised batch Poisson GLM)
# ===========================================================================

def mum_pvalues(Z_preprocessed: np.ndarray, Y: np.ndarray,
                max_iter: int = 50, tol: float = 1e-8,
                age_col: int = 0):
    """Voxelwise Poisson log-link GLM via vectorised batch IRLS.

    Tests H0: age coefficient == 0 at every voxel simultaneously.

    Parameters
    ----------
    Z_preprocessed : (n, n_cov) preprocessed design matrix [cov_0, ..., intercept]
                     — intercept is last column.
    age_col        : int
        Column index of the age covariate in Z_preprocessed.
        0 for GRF simulated data, 1 for UKB real data (sex is column 0).
    Y              : (n, V) count observations.
    max_iter, tol  : IRLS convergence controls.

    Returns
    -------
    p_vals  : (V,) two-sided Wald p-values for the age coefficient.
    z_stats : (V,) corresponding z-statistics.
    """
    Z = Z_preprocessed.astype(np.float64)
    Y = Y.astype(np.float64)
    n, n_cov = Z.shape
    V        = Y.shape[1]

    # Initialise beta: intercept ← log(max(mean_y, ε)), others ← 0
    beta = np.zeros((n_cov, V), dtype=np.float64)
    beta[-1] = np.log(np.maximum(Y.mean(axis=0), 1e-10))

    for _ in range(max_iter):
        eta = Z @ beta          # (n, V)
        mu  = np.exp(np.clip(eta, -30, 30))    # (n, V)
        resid = Y - mu          # (n, V)  score numerator components

        # Fisher information (n_cov x n_cov x V): ZtWZ[r,k,v] = Σ_i Z[i,r] Z[i,k] mu[i,v]
        ZtWZ = np.einsum("ir,ik,iv->rkv", Z, Z, mu)  # (n_cov, n_cov, V)

        # Newton step:  delta = ZtWZ^{-1} @ (Z^T resid)
        score = Z.T @ resid     # (n_cov, V)

        # Solve n_cov×n_cov linear system per voxel.
        # For n_cov=2 use analytic inverse; otherwise use batch solve.
        if n_cov == 2:
            a = ZtWZ[0, 0]; b = ZtWZ[0, 1]; d = ZtWZ[1, 1]
            det = a * d - b * b + 1e-15
            delta = np.empty_like(score)
            delta[0] =  (d * score[0] - b * score[1]) / det
            delta[1] = (-b * score[0] + a * score[1]) / det
        else:
            # (n_cov, n_cov, V) → (V, n_cov, n_cov) for np.linalg.solve
            lhs = ZtWZ.transpose(2, 0, 1) + 1e-12 * np.eye(n_cov)[None]
            rhs = score.T[:, :, None]       # (V, n_cov, 1)
            delta = np.linalg.solve(lhs, rhs).squeeze(-1).T  # (n_cov, V)

        beta += delta
        if np.max(np.abs(score)) < tol:
            break

    # Wald test for age coefficient (index age_col)
    eta = Z @ beta
    mu  = np.exp(np.clip(eta, -30, 30))
    ZtWZ = np.einsum("ir,ik,iv->rkv", Z, Z, mu)  # (n_cov, n_cov, V)

    lhs  = ZtWZ.transpose(2, 0, 1) + 1e-12 * np.eye(n_cov)[None]  # (V, n_cov, n_cov)
    eye_k = np.zeros(n_cov); eye_k[age_col] = 1.0
    rhs  = np.tile(eye_k, (V, 1))[:, :, None]
    col_k = np.linalg.solve(lhs, rhs).squeeze(-1)                  # (V, n_cov)
    var_age = col_k[:, age_col]

    se_age  = np.sqrt(np.maximum(var_age, 1e-12))
    z_stats = beta[age_col] / se_age
    p_vals  = 2.0 * scipy.stats.norm.sf(np.abs(z_stats))
    return p_vals.ravel(), z_stats.ravel()


# ===========================================================================
# Reference prediction map (fitted on full data)
# ===========================================================================

def compute_reference_map(Z_preprocessed: np.ndarray, Y: np.ndarray,
                           X_spatial: np.ndarray,
                           data_dict: dict, marginal_dist: str,
                           link_func: str, device: torch.device) -> np.ndarray:
    """Fit S-GLM on full data; return predicted mean P_mean (N_voxels,)."""
    params = fit_sglm(data_dict, marginal_dist, link_func, device)
    BR = BrainRegression_Approximate.__new__(BrainRegression_Approximate)
    BR.simulated_dset = True
    BR.dtype  = torch.float64
    BR.device = device
    BR.load_data(data_dict, "SpatialBrainLesion")
    _, _, P_mean = BR.goodness_of_fit(beta=params["beta"],
                                      model="SpatialBrainLesion",
                                      mode="dask", block_size=5000)
    gc.collect()
    return P_mean.ravel()


def compute_accuracy(P_sub: np.ndarray, P_ref: np.ndarray):
    """Return (RMSE, Pearson correlation) between prediction maps."""
    diff = P_sub - P_ref
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    corr = float(np.corrcoef(P_sub, P_ref)[0, 1])
    return rmse, corr


# ===========================================================================
# FPR helpers
# ===========================================================================

def compute_fpr(p_vals: np.ndarray, alpha: float = 0.05,
                apply_fdr: bool = False) -> float:
    """Compute FPR = (# significant voxels) / (total voxels)."""
    if apply_fdr:
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(p_vals, alpha=alpha, method="fdr_bh")
        sig = np.sum(p_adj < alpha)
    else:
        sig = int(np.sum(p_vals < alpha))
    return sig / len(p_vals)


# ===========================================================================
# Single repetition worker (runs in parallel)
# ===========================================================================

def run_one_rep(rep: int, N: int, base_data: dict,
                args: argparse.Namespace, master_seed: int) -> dict:
    """Run one subsampling + permutation repetition.

    Returns a results dict with all metrics for this (N, rep) combination.
    """
    rng  = np.random.default_rng(master_seed)
    device = torch.device("cpu")

    simulated_dset = not args.use_ukb
    age_col        = base_data.get("age_col", 0)  # 0 for GRF, 2 for UKB

    Z_raw     = base_data["Z"]        # (n_full, C) raw covariates
    Y_full    = base_data["Y"]        # (n_full, V)
    X_spatial = base_data["X_spatial"]  # (V, P)

    # -----------------------------------------------------------------------
    # 1. Stratified subsample
    # -----------------------------------------------------------------------
    idx = stratified_subsample(Z_raw, n_bins=args.n_age_bins,
                                n_sample=N, rng=rng, age_col=age_col)
    Z_sub_raw = Z_raw[idx]          # (N, C) raw
    Y_sub     = Y_full[idx]         # (N, V)

    # Preprocess Z (standardise age, add polynomial terms)
    Z_sub_pre = preprocess_Z(simulated_dset=simulated_dset, Z=Z_sub_raw,
                             polynomial_order=args.polynomial_order)
    # Z_sub_pre: (N, polynomial_order)  — no intercept yet (added inside load_data)

    data_sub = make_multigroup_data(Z_sub_raw[:, 1:], Y_sub, X_spatial)
    # regression.py load_data expects Z without subjectID (col 0 stripped);
    # inference.py load_params uses preprocess_Z output.
    data_sub_for_inference = make_multigroup_data(Z_sub_pre, Y_sub, X_spatial)

    # For UKB preprocessed Z = [Sex, Age, Headsize, CVR] → age is at index 1.
    # For GRF preprocessed Z = [age_std, ...]              → age is at index 0.
    age_col_preprocessed = 0 if simulated_dset else 1

    result = {"N": N, "rep": rep}

    # -----------------------------------------------------------------------
    # 2. Fit S-GLM on real subsample  (data_sub uses raw Z for regression)
    # -----------------------------------------------------------------------
    sglm_params = fit_sglm(data_sub, args.marginal_dist,
                            args.link_func, device, simulated_dset=simulated_dset)
    try:
        p_sglm_real, z_sglm_real = sglm_pvalues(
            data_sub_for_inference, sglm_params,
            args.inference_method, args.polynomial_order, device,
            simulated_dset=simulated_dset,
            age_col_preprocessed=age_col_preprocessed,
        )
        result["p_sglm_real"] = p_sglm_real.astype(np.float32)
        result["z_sglm_real"] = z_sglm_real.astype(np.float32)
        del p_sglm_real, z_sglm_real
    except Exception as exc:
        logger.warning("N=%d rep=%d: S-GLM real-data inference failed — %s", N, rep, exc)
        result["p_sglm_real"] = None

    # -----------------------------------------------------------------------
    # 3. Fit MUM on real subsample (inference on real data)
    # -----------------------------------------------------------------------
    # MUM operates on preprocessed Z (with intercept appended)
    intercept_col = np.ones((Z_sub_pre.shape[0], 1))
    Z_sub_mum = np.concatenate([Z_sub_pre, intercept_col], axis=1)
    try:
        p_mum_real, z_mum_real = mum_pvalues(Z_sub_mum, Y_sub.astype(np.float64),
                                              age_col=age_col_preprocessed)
        result["p_mum_real"] = p_mum_real.astype(np.float32)
        result["z_mum_real"] = z_mum_real.astype(np.float32)
        del p_mum_real, z_mum_real
    except Exception as exc:
        logger.warning("N=%d rep=%d: MUM real-data inference failed — %s", N, rep, exc)
        result["p_mum_real"] = None

    # -----------------------------------------------------------------------
    # 4. Accuracy vs reference (compute P_mean from S-GLM fit)
    # -----------------------------------------------------------------------
    try:
        BR_sub = BrainRegression_Approximate(simulated_dset=simulated_dset,
                                              dtype=torch.float64, device=device)
        BR_sub.load_data(data_sub, "SpatialBrainLesion")
        _, _, P_sub = BR_sub.goodness_of_fit(
            beta=sglm_params["beta"],
            model="SpatialBrainLesion",
            mode="dask", block_size=5000,
        )
        del BR_sub; gc.collect()
        result["P_sub_sglm"] = P_sub.ravel().astype(np.float32)
        del P_sub
    except Exception as exc:
        logger.warning("N=%d rep=%d: goodness_of_fit failed — %s", N, rep, exc)
        result["P_sub_sglm"] = None

    # -----------------------------------------------------------------------
    # 5. NULL TEST — permute age within strata and run inference
    # -----------------------------------------------------------------------
    # Build a single consistent permutation used by both S-GLM and MUM.
    perm_order = rng.permutation(len(idx))
    Z_perm_raw = Z_sub_raw[perm_order]           # permuted raw covariates
    Z_perm_pre = preprocess_Z(simulated_dset=simulated_dset, Z=Z_perm_raw,
                               polynomial_order=args.polynomial_order)
    data_perm_reg  = make_multigroup_data(Z_perm_raw[:, 1:], Y_sub, X_spatial)
    data_perm_inf  = make_multigroup_data(Z_perm_pre, Y_sub, X_spatial)

    # S-GLM fit on permuted data
    try:
        sglm_perm_params = fit_sglm(data_perm_reg, args.marginal_dist,
                                     args.link_func, device,
                                     simulated_dset=simulated_dset)
        del data_perm_reg; gc.collect()
        p_sglm, z_sglm = sglm_pvalues(data_perm_inf, sglm_perm_params,
                                        args.inference_method,
                                        args.polynomial_order, device,
                                        simulated_dset=simulated_dset,
                                        age_col_preprocessed=age_col_preprocessed)
        del data_perm_inf; del sglm_perm_params; gc.collect()
        result["p_sglm"] = p_sglm.astype(np.float32)
        result["z_sglm"] = z_sglm.astype(np.float32)
        result["fpr_sglm"] = float(compute_fpr(
            p_sglm, args.alpha_threshold, apply_fdr=False))
        if args.fdr:
            result["fpr_sglm_fdr"] = float(compute_fpr(
                p_sglm, args.alpha_threshold, apply_fdr=True))
        del p_sglm, z_sglm
    except Exception as exc:
        logger.warning("N=%d rep=%d: S-GLM inference failed — %s", N, rep, exc)
        result["p_sglm"]     = None
        result["fpr_sglm"]   = float("nan")

    # MUM fit on permuted data (vectorised IRLS, no external regression class)
    try:
        intercept_perm = np.ones((Z_perm_pre.shape[0], 1))
        Z_perm_mum = np.concatenate([Z_perm_pre, intercept_perm], axis=1)
        p_mum, z_mum = mum_pvalues(Z_perm_mum, Y_sub.astype(np.float64),
                                    age_col=age_col_preprocessed)
        del Z_perm_mum; gc.collect()
        result["p_mum"] = p_mum.astype(np.float32)
        result["z_mum"] = z_mum.astype(np.float32)
        result["fpr_mum"] = float(compute_fpr(
            p_mum, args.alpha_threshold, apply_fdr=False))
        if args.fdr:
            result["fpr_mum_fdr"] = float(compute_fpr(
                p_mum, args.alpha_threshold, apply_fdr=True))
        del p_mum, z_mum
    except Exception as exc:
        logger.warning("N=%d rep=%d: MUM inference failed — %s", N, rep, exc)
        result["p_mum"]   = None
        result["fpr_mum"] = float("nan")

    logger.info(
        "N=%4d  rep=%3d  FPR_SGLM=%.4f  FPR_MUM=%.4f",
        N, rep,
        result.get("fpr_sglm", float("nan")),
        result.get("fpr_mum",  float("nan")),
    )
    return result


# ===========================================================================
# Aggregation and saving
# ===========================================================================

def aggregate_metrics(all_results: list, args: argparse.Namespace) -> list[dict]:
    """Aggregate raw results into per-(N, method) summary rows."""
    from collections import defaultdict

    rows = []
    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        N = r["N"]
        for method in ("sglm", "mum"):
            fpr_key = f"fpr_{method}"
            if fpr_key in r and not np.isnan(r[fpr_key]):
                grouped[N][method].append(r[fpr_key])

    for N in sorted(grouped):
        for method, fprs in grouped[N].items():
            row = {
                "N": N,
                "method": method.upper(),
                "fpr_mean": float(np.mean(fprs)),
                "fpr_std": float(np.std(fprs)),
                "fpr_min": float(np.min(fprs)),
                "fpr_max": float(np.max(fprs)),
                "n_reps":  len(fprs),
            }
            rows.append(row)
    return rows


def save_one_rep(result: dict, ts: str):
    """Save a single repetition's results immediately after it completes.

    File: raw/rep_N{N}_rep{rep}_seed{seed}.npz  (when seed is available)
          raw/rep_N{N}_rep{rep}_{ts}.npz        (fallback)
    Safe to call from worker processes or the main process.
    """
    N    = result["N"]
    rep  = result["rep"]
    seed = result.get("seed")
    if seed is not None:
        out_path = os.path.join(RAW_DIR, f"rep_N{N}_rep{rep}_seed{seed}.npz")
    else:
        out_path = os.path.join(RAW_DIR, f"rep_N{N}_rep{rep}_{ts}.npz")
    payload = {}
    for key in ("p_sglm", "z_sglm", "p_mum", "z_mum", "P_sub_sglm",
                 "p_mum_real", "z_mum_real", "p_sglm_real", "z_sglm_real"):
        val = result.get(key)
        if val is not None:
            payload[key] = val
    for key in ("fpr_sglm", "fpr_mum", "fpr_sglm_fdr", "fpr_mum_fdr"):
        if key in result:
            payload[key] = np.array(result[key])
    payload["N"]   = np.array(N)
    payload["rep"] = np.array(rep)
    if seed is not None:
        payload["seed"] = np.array(seed)
    np.savez_compressed(out_path, **payload)
    logger.info("Saved rep result  → %s", out_path)


def save_raw_results(all_results: list, N: int, ts: str):
    """Save per-rep raw p-value arrays for a given N."""
    out_path = os.path.join(RAW_DIR, f"raw_N{N}_{ts}.npz")
    payload = {}
    for r in all_results:
        if r["N"] != N:
            continue
        rep = r["rep"]
        for key in ("p_sglm", "z_sglm", "p_mum", "z_mum", "P_sub_sglm",
                     "p_mum_real", "z_mum_real", "p_sglm_real", "z_sglm_real"):
            val = r.get(key)
            if val is not None:
                payload[f"{key}_rep{rep}"] = val
        for key in ("fpr_sglm", "fpr_mum"):
            if key in r:
                payload[f"{key}_rep{rep}"] = np.array(r[key])
    np.savez_compressed(out_path, **payload)
    logger.info("Saved raw results → %s", out_path)


def save_metrics_csv(rows: list, ts: str):
    """Write aggregated metrics to CSV."""
    csv_path = os.path.join(METRICS_DIR, f"metrics_{ts}.csv")
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved metrics CSV → %s", csv_path)
    return csv_path


# ===========================================================================
# PP-plot generation
# ===========================================================================

def _pp_curve(ax, p_list, colour, label):
    """Helper: plot a single PP-curve on ax from a list of p-value arrays."""
    if not p_list:
        return
    p_cat = np.concatenate(p_list)
    p_cat = p_cat[np.isfinite(p_cat)]
    p_sorted = np.sort(p_cat)
    n_pts    = len(p_sorted)
    expected = np.linspace(0, 1, n_pts)
    ax.plot(expected, p_sorted, color=colour, lw=1.2,
            label=f"{label}  (n={n_pts:,})", alpha=0.85)


def make_pp_plot(all_results: list, N: int, args: argparse.Namespace, ts: str):
    """Generate and save PP-plots for a given N.

    Produces two side-by-side panels:
      Left  — permuted-age (null) p-values: S-GLM vs MUM
      Right — real-data p-values: S-GLM (from sglm_pvalues on real data) vs MUM
    """
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

    if not p_sglm_null and not p_mum_null and not p_mum_real:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # --- Left panel: null (permuted age) ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal (uniform)")
    _pp_curve(ax, p_sglm_null, "#E64646", "S-GLM (null)")
    _pp_curve(ax, p_mum_null,  "#4682B4", "MUM (null)")
    ax.set_xlabel("Expected quantile (Uniform)")
    ax.set_ylabel("Observed p-value quantile")
    ax.set_title(f"PP-plot — AGE (null, permuted)  |  N={N}  R={args.R}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # --- Right panel: real data ---
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal (uniform)")
    _pp_curve(ax, p_sglm_real, "#E64646", "S-GLM (real)")
    _pp_curve(ax, p_mum_real,  "#4682B4", "MUM (real)")
    ax.set_xlabel("Expected quantile (Uniform)")
    ax.set_ylabel("Observed p-value quantile")
    ax.set_title(f"PP-plot — AGE (real data)  |  N={N}  R={args.R}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"PPplot_N{N}_{ts}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved PP-plot → %s", out_path)


def make_fpr_summary_plot(rows: list, alpha: float, ts: str):
    """Bar chart of mean FPR ± std for S-GLM vs MUM across N."""
    if not rows:
        return
    N_vals   = sorted(set(r["N"] for r in rows))
    methods  = ["SGLM", "MUM"]
    colours  = {"SGLM": "#E64646", "MUM": "#4682B4"}
    x        = np.arange(len(N_vals))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, method in enumerate(methods):
        means = []
        stds  = []
        for N in N_vals:
            row = next((r for r in rows
                        if r["N"] == N and r["method"] == method), None)
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

    out_path = os.path.join(PLOTS_DIR, f"FPR_summary_{ts}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved FPR summary plot → %s", out_path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = get_args()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Single-seed mode: treat --seed as the master seed for one realization.
    single_seed_mode = args.seed is not None

    # ------------------------------------------------------------------
    # Resolve output directories (depends on R from args)
    # ------------------------------------------------------------------
    _setup_output_dirs(args.R, use_ukb=args.use_ukb)

    # Add file log handler now that LOGS_DIR is known
    file_handler = logging.FileHandler(
        os.path.join(LOGS_DIR, f"experiment_{ts}.log")
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # Log experiment config
    config = vars(args)
    config["timestamp"]    = ts
    config["experiment_dir"] = EXPERIMENT_OUT
    cfg_path = os.path.join(LOGS_DIR, f"config_{ts}.json")
    with open(cfg_path, "w") as fh:
        json.dump(config, fh, indent=2)
    logger.info("Experiment config: %s", json.dumps(config, indent=2))

    # ------------------------------------------------------------------
    # Load base (full) dataset
    # ------------------------------------------------------------------
    if args.use_ukb:
        logger.info("Loading UKB real dataset (spacing=%d) …", args.spacing)
        base_data = load_ukb_data(spacing=args.spacing)
        logger.info("UKB data loaded: n=%d, V=%d, P=%d",
                    base_data["Z"].shape[0], base_data["Y"].shape[1],
                    base_data["X_spatial"].shape[1])
    else:
        logger.info("Loading GRF base data (n_subject=%d, seed=%d) …",
                    args.data_n_subject, args.data_seed)
        base_data = load_base_data(args.data_n_subject, args.data_seed)
        logger.info("GRF data loaded: n=%d, V=%d, P=%d",
                    base_data["Z"].shape[0], base_data["Y"].shape[1],
                    base_data["X_spatial"].shape[1])
    n_full    = base_data["Z"].shape[0]
    n_voxels  = base_data["Y"].shape[1]

    all_results = []

    # ------------------------------------------------------------------
    # Main loop over subsample sizes
    # ------------------------------------------------------------------
    for N in args.N_list:
        if N > n_full:
            logger.warning(
                "N=%d > n_full=%d — clamping to full dataset size.", N, n_full)
            N = n_full

        logger.info("=" * 60)
        logger.info("Subsample size N=%d  (R=%d repetitions)", N, args.R)
        logger.info("=" * 60)

        t0 = time.time()

        rep_results = []

        if single_seed_mode:
            # ----------------------------------------------------------
            # Single-realization mode: run exactly ONE rep with the
            # specified --seed.  --rep labels the repetition index for
            # output file naming.
            # ----------------------------------------------------------
            logger.info("Single-seed mode: seed=%d, rep=%d",
                        args.seed, args.rep)
            res = run_one_rep(rep=args.rep, N=N, base_data=base_data,
                              args=args, master_seed=args.seed)
            res["seed"] = args.seed
            save_one_rep(res, ts)
            rep_results.append(res)
        else:
            # ----------------------------------------------------------
            # Multi-realization mode: loop over R reps, optionally parallel
            # ----------------------------------------------------------
            # Each rep gets a unique seed derived from base_seed, N, and rep index
            seeds = [args.base_seed + N * 1000 + r for r in range(args.R)]

            n_jobs = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
            worker = functools.partial(run_one_rep, N=N, base_data=base_data,
                                       args=args)

            if n_jobs == 1:
                # Sequential fallback — useful for debugging
                for r in range(args.R):
                    res = worker(rep=r, master_seed=seeds[r])
                    save_one_rep(res, ts)
                    rep_results.append(res)
            else:
                with concurrent.futures.ProcessPoolExecutor(
                        max_workers=n_jobs) as executor:
                    futures = {
                        executor.submit(worker, rep=r, master_seed=seeds[r]): r
                        for r in range(args.R)
                    }
                    for fut in concurrent.futures.as_completed(futures):
                        try:
                            res = fut.result()
                            save_one_rep(res, ts)
                            rep_results.append(res)
                        except Exception as exc:
                            r = futures[fut]
                            logger.error(
                                "N=%d rep=%d raised an exception: %s", N, r, exc)

        all_results.extend(rep_results)
        save_raw_results(rep_results, N, ts)
        make_pp_plot(rep_results, N, args, ts)

        elapsed = time.time() - t0
        fpr_sglm_vals = [r["fpr_sglm"] for r in rep_results
                         if "fpr_sglm" in r and not np.isnan(r["fpr_sglm"])]
        fpr_mum_vals  = [r["fpr_mum"]  for r in rep_results
                         if "fpr_mum"  in r and not np.isnan(r["fpr_mum"])]
        logger.info(
            "N=%d done in %.1fs  |  FPR_SGLM=%.4f\u00b1%.4f  FPR_MUM=%.4f\u00b1%.4f",
            N, elapsed,
            np.mean(fpr_sglm_vals) if fpr_sglm_vals else float("nan"),
            np.std(fpr_sglm_vals)  if fpr_sglm_vals else 0.0,
            np.mean(fpr_mum_vals)  if fpr_mum_vals  else float("nan"),
            np.std(fpr_mum_vals)   if fpr_mum_vals  else 0.0,
        )
        del rep_results; gc.collect()

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    rows     = aggregate_metrics(all_results, args)
    csv_path = save_metrics_csv(rows, ts)
    make_fpr_summary_plot(rows, args.alpha_threshold, ts)

    # Print final summary table
    logger.info("\n%s", "-" * 62)
    logger.info("%-8s  %-8s  %10s  %10s  %6s", "N", "Method",
                "FPR mean", "FPR std", "n_reps")
    logger.info("%s", "-" * 62)
    for row in rows:
        logger.info("%-8d  %-8s  %10.4f  %10.4f  %6d",
                    row["N"], row["method"],
                    row["fpr_mean"], row["fpr_std"], row["n_reps"])
    logger.info("%s", "-" * 62)

    logger.info("Experiment complete.  Outputs in: %s", EXPERIMENT_OUT)


if __name__ == "__main__":
    main()
