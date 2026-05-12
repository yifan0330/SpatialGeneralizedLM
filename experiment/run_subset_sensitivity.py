"""
Run subset-sensitivity experiments on UKB lesion data.

This script repeatedly calls run.py with different random seeds for selecting
50 UKB subjects, fits both models, runs inference, and summarizes sensitivity
and stability of the resulting p-value / z-score maps.

Example
-------
python run_subset_sensitivity.py --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import nibabel as nib
from scipy.stats import norm

from plot import plot_brain


DEFAULT_MODELS = ["SpatialBrainLesion", "MassUnivariateRegression"]
MODEL_TO_METHOD = {
    "SpatialBrainLesion": "S-GLM",
    "MassUnivariateRegression": "MUM",
}
METHOD_TO_SUBSAMPLING_KEY = {
    "S-GLM": "z_sglm_real",
    "MUM": "z_mum_real",
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UKB subset sensitivity/stability experiments."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="Random seeds used to choose different UKB subject subsets.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Models to run for each random seed.",
    )
    parser.add_argument("--UKB_subject", type=int, default=50)
    parser.add_argument("--spacing", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--inference_method", default="sandwich", choices=["FI", "sandwich"])
    parser.add_argument("--gpus", default="0")
    parser.add_argument(
        "--plot_z_maps",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Plot per-seed age-effect Z-maps and cross-seed significant-voxel frequency maps.",
    )
    parser.add_argument(
        "--zmap_dir",
        default=None,
        help="Directory for sensitivity Z-map plots (default: results/UKB_<N>/subset_sensitivity/zmap_plots_<timestamp>).",
    )
    parser.add_argument(
        "--z_vmax",
        type=float,
        default=None,
        help="Symmetric colour scale limit for signed Z-maps. Default uses the 99th percentile of |Z|.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop after the first failed run.py command.",
    )
    parser.add_argument(
        "--skip_runs",
        action="store_true",
        help="Skip calling run.py and only summarize/plot existing results.",
    )
    parser.add_argument(
        "--plot_combined_z_maps",
        type=lambda x: x.lower() == "true",
        default=False,
        help=(
            "Create one Z-map figure per N from existing subsampling results. "
            "Each figure has two rows, S-GLM and MUM, with one column per requested seed. "
            "The requested --seeds values are interpreted as subsampling repetition indices "
            "when --skip_runs is used."
        ),
    )
    parser.add_argument(
        "--subsampling_raw_dir",
        default=None,
        help=(
            "Raw rep_*.npz directory from run_subsampling_experiment.py. Default: "
            "experiments/subsampling_experiment_UKB_R100/results/raw."
        ),
    )
    parser.add_argument(
        "--combined_N_list",
        nargs="+",
        type=int,
        default=None,
        help="Subset of N values to include in the combined Z-map figure.",
    )
    parser.add_argument(
        "--combined_z_slice",
        type=int,
        default=18,
        help="MNI/world z coordinate for the combined Z-map figure; default matches cut_coords z=18.",
    )
    parser.add_argument(
        "--combined_zmap_output",
        default=None,
        help="Output PNG path for the combined Z-map figure.",
    )
    parser.add_argument(
        "--include_sensitivity_frequency",
        type=lambda x: x.lower() == "true",
        default=True,
        help=(
            "Add a final per-row panel showing cross-seed significant-voxel "
            "frequency from results/UKB_<N>/subset_sensitivity."
        ),
    )
    parser.add_argument(
        "--sensitivity_frequency_cmap",
        default="viridis",
        help="Colourmap for sensitivity-frequency panels; Z-map panels keep inferno.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, model: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "run.py",
        "--n_auxiliary=0",
        "--simulated_dset", "False",
        "--UKB_subject", str(args.UKB_subject),
        "--model", model,
        "--regression_terms", "multiplicative", "additive",
        "--spacing", str(args.spacing),
        "--space_dim", "brain",
        "--gradient_mode", "dask",
        "--preconditioner_mode", "approximate",
        "--contrast_name", "age",
        "--full_model", "False",
        "--marginal_dist", "Poisson",
        "--link_func", "log",
        "--polynomial_order", "1",
        "--firth_penalty", "False",
        "--run_data_generation", "False",
        "--run_regression", "True",
        "--run_inference", "True",
        "--inference_method", args.inference_method,
        "--random_seed", str(seed),
        "--gpus", args.gpus,
    ]


def _load_first_array(npz_path: str, preferred_key: str | None = None) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as data:
        if preferred_key is not None and preferred_key in data.files:
            return np.asarray(data[preferred_key]).astype(float).ravel()
        for key in data.files:
            arr = np.asarray(data[key])
            if arr.size > 0:
                return arr.astype(float).ravel()
    return np.array([], dtype=float)


def _find_result_file(results_dir: Path, prefix: str, model: str, seed: int,
                      inference_method: str) -> str | None:
    pattern = str(
        results_dir
        / f"{prefix}_{model}*random_seed_{seed}*{inference_method}.npz"
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def archive_run_outputs(source_dir: Path, target_dir: Path, model: str, seed: int) -> list[str]:
    """Move run-specific output files into the subset_sensitivity folder."""
    target_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        f"Regression_*{model}*random_seed_{seed}*.npz",
        f"Inference_*{model}*random_seed_{seed}*.npz",
        f"XTWX_*{model}*random_seed_{seed}*.npz",
        f"Fisher_info_*{model}*random_seed_{seed}*.npz",
        f"meat_term_*{model}*random_seed_{seed}*.npz",
        f"bread_term_*{model}*random_seed_{seed}*.npz",
        f"p_values_*{model}*random_seed_{seed}*.npz",
        f"z_values_*{model}*random_seed_{seed}*.npz",
    ]

    moved = []
    for pattern in patterns:
        for src in source_dir.glob(pattern):
            dst = target_dir / src.name
            if src.resolve() == dst.resolve():
                continue
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            moved.append(str(dst))
    return moved


def summarize_one(results_dir: Path, model: str, seed: int, alpha: float,
                  inference_method: str) -> dict:
    p_file = _find_result_file(results_dir, "p_values", model, seed, inference_method)
    z_file = _find_result_file(results_dir, "z_values", model, seed, inference_method)

    row = {
        "model": model,
        "seed": seed,
        "p_file": p_file or "",
        "z_file": z_file or "",
        "n_voxels": 0,
        "rejection_rate": np.nan,
        "min_p": np.nan,
        "median_p": np.nan,
        "max_abs_z": np.nan,
    }

    if p_file:
        p = _load_first_array(p_file, preferred_key="p_vals")
        p = p[np.isfinite(p)]
        p = p[(p >= 0) & (p <= 1)]
        row["n_voxels"] = int(p.size)
        if p.size:
            row["rejection_rate"] = float(np.mean(p < alpha))
            row["min_p"] = float(np.min(p))
            row["median_p"] = float(np.median(p))

    if z_file:
        z = _load_first_array(z_file, preferred_key="z_stats")
        z = z[np.isfinite(z)]
        if z.size:
            row["max_abs_z"] = float(np.max(np.abs(z)))

    return row


def summarize_pairwise_stability(results_dir: Path, models: list[str], seeds: list[int],
                                 alpha: float, inference_method: str) -> list[dict]:
    rows = []
    for model in models:
        p_by_seed = {}
        z_by_seed = {}
        for seed in seeds:
            p_file = _find_result_file(results_dir, "p_values", model, seed, inference_method)
            z_file = _find_result_file(results_dir, "z_values", model, seed, inference_method)
            if p_file:
                p_by_seed[seed] = _load_first_array(p_file, preferred_key="p_vals")
            if z_file:
                z_by_seed[seed] = _load_first_array(z_file, preferred_key="z_stats")

        for seed_a, seed_b in combinations(seeds, 2):
            p_a = p_by_seed.get(seed_a)
            p_b = p_by_seed.get(seed_b)
            z_a = z_by_seed.get(seed_a)
            z_b = z_by_seed.get(seed_b)

            row = {
                "model": model,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "z_correlation": np.nan,
                "discoveries_a": np.nan,
                "discoveries_b": np.nan,
                "discovery_jaccard": np.nan,
            }

            if z_a is not None and z_b is not None:
                m = min(z_a.size, z_b.size)
                za = z_a[:m]
                zb = z_b[:m]
                mask = np.isfinite(za) & np.isfinite(zb)
                if np.sum(mask) > 1:
                    row["z_correlation"] = float(np.corrcoef(za[mask], zb[mask])[0, 1])

            if p_a is not None and p_b is not None:
                m = min(p_a.size, p_b.size)
                da = np.isfinite(p_a[:m]) & (p_a[:m] < alpha)
                db = np.isfinite(p_b[:m]) & (p_b[:m] < alpha)
                union = np.sum(da | db)
                row["discoveries_a"] = int(np.sum(da))
                row["discoveries_b"] = int(np.sum(db))
                row["discovery_jaccard"] = float(np.sum(da & db) / union) if union else 1.0

            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_matching_brain_mask(project_dir: Path, expected_voxels: int) -> nib.Nifti1Image:
    """Load the lesion/brain mask whose positive voxels match a Z-map length."""
    candidates = [
        project_dir / "data" / "UKB" / "smooth_lesion_mask_RealDataset.nii.gz",
        project_dir / "data" / "brain" / "smooth_lesion_mask_Simulation.nii.gz",
        project_dir / "data" / "UKB" / "lesion_mask_RealDataset.nii.gz",
        project_dir.parent / "GRF_data" / "MNI152_T1_2mm_brain_mask.nii.gz",
    ]
    existing = [path for path in candidates if path.exists()]
    for path in existing:
        mask = nib.load(str(path))
        n_mask_voxels = int(np.sum(mask.get_fdata() > 0))
        if n_mask_voxels == expected_voxels:
            return mask

    checked = ", ".join(str(p) for p in existing) if existing else "none found"
    raise ValueError(
        f"No mask with {expected_voxels} positive voxels found. "
        f"Checked: {checked}"
    )


def _parse_subsampling_rep_file(path: Path) -> tuple[int, int, int] | None:
    """Parse rep_N{N}_rep{rep}_seed{seed}.npz filenames."""
    try:
        parts = path.stem.split("_")
        N = int(parts[1][1:])
        rep = int(parts[2][3:])
        seed = int(parts[3][4:])
        return N, rep, seed
    except (IndexError, ValueError):
        return None


def _finite_mean_stack(arrays: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Mean across arrays without warning when some voxels are missing in every array."""
    min_len = min(a.size for a in arrays)
    stack = np.vstack([a[:min_len] for a in arrays])
    finite = np.isfinite(stack)
    counts = np.sum(finite, axis=0)
    sums = np.sum(np.where(finite, stack, 0.0), axis=0)
    mean = np.full(min_len, np.nan, dtype=float)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return mean, stack.shape[0]


def _mean_subsampling_zmaps(raw_dir: Path, N_list: list[int] | None) -> dict[str, dict[int, tuple[np.ndarray, int]]]:
    """Average real-data subsampling Z-maps across all random seeds for each N."""
    wanted = set(N_list) if N_list is not None else None
    grouped: dict[str, dict[int, list[np.ndarray]]] = {
        "S-GLM": {},
        "MUM": {},
    }

    for path in sorted(raw_dir.glob("rep_N*_rep*_seed*.npz")):
        parsed = _parse_subsampling_rep_file(path)
        if parsed is None:
            continue
        N, _, _ = parsed
        if wanted is not None and N not in wanted:
            continue

        with np.load(path, allow_pickle=False) as data:
            for method, key in METHOD_TO_SUBSAMPLING_KEY.items():
                if key not in data.files:
                    continue
                z = np.asarray(data[key], dtype=float).ravel()
                if z.size:
                    grouped[method].setdefault(N, []).append(z)

    means: dict[str, dict[int, tuple[np.ndarray, int]]] = {"S-GLM": {}, "MUM": {}}
    for method, by_N in grouped.items():
        for N, arrays in by_N.items():
            if not arrays:
                continue
            means[method][N] = _finite_mean_stack(arrays)
    return means


def _load_subsampling_zmaps_by_rep(raw_dir: Path, N_list: list[int] | None,
                                   reps: list[int]) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    """Load real-data S-GLM/MUM Z-maps by N and repetition index."""
    wanted_N = set(N_list) if N_list is not None else None
    wanted_reps = set(reps)
    out: dict[int, dict[int, dict[str, np.ndarray]]] = {}

    for path in sorted(raw_dir.glob("rep_N*_rep*_seed*.npz")):
        parsed = _parse_subsampling_rep_file(path)
        if parsed is None:
            continue
        N, rep, _ = parsed
        if wanted_N is not None and N not in wanted_N:
            continue
        if rep not in wanted_reps:
            continue

        with np.load(path, allow_pickle=False) as data:
            entry = out.setdefault(N, {}).setdefault(rep, {})
            for method, key in METHOD_TO_SUBSAMPLING_KEY.items():
                if key in data.files:
                    z = np.asarray(data[key], dtype=float).ravel()
                    if z.size:
                        entry[method] = z
    return out


def _mean_sensitivity_zmaps(results_dir: Path, models: list[str], seeds: list[int],
                            inference_method: str) -> dict[str, tuple[np.ndarray, int]]:
    """Average subset-sensitivity Z-maps across the requested seeds."""
    means = {}
    for model in models:
        method = MODEL_TO_METHOD.get(model, model)
        arrays = []
        for seed in seeds:
            z_file = _find_result_file(results_dir, "z_values", model, seed, inference_method)
            if z_file is None:
                continue
            z = _load_first_array(z_file, preferred_key="z_stats")
            if z.size:
                arrays.append(z)
        if arrays:
            means[method] = _finite_mean_stack(arrays)
    return means


def _sensitivity_frequency_for_N(project_dir: Path, N: int, models: list[str],
                                 reps: list[int], inference_method: str,
                                 alpha: float) -> dict[str, tuple[np.ndarray, int]]:
    """Return significant-voxel frequency maps for sensitivity runs at one N."""
    sens_dir = project_dir / "results" / f"UKB_{N}" / "subset_sensitivity"
    if not sens_dir.is_dir():
        return {}

    z_threshold = float(norm.ppf(1.0 - alpha))
    frequency_maps = {}
    for model in models:
        method = MODEL_TO_METHOD.get(model, model)
        masks = []
        for rep in reps:
            z_file = _find_result_file(sens_dir, "z_values", model, rep, inference_method)
            if z_file is None:
                continue
            z = _load_first_array(z_file, preferred_key="z_stats")
            if z.size:
                masks.append((np.isfinite(z) & (np.abs(z) >= z_threshold)).astype(float))
        if masks:
            min_len = min(mask.size for mask in masks)
            stack = np.vstack([mask[:min_len] for mask in masks])
            frequency_maps[method] = (np.mean(stack, axis=0), stack.shape[0])
    return frequency_maps


def _crop_white_border(img: np.ndarray, pad: int = 4) -> np.ndarray:
    """Crop near-white margins from a rendered nilearn panel image."""
    if img.ndim < 3:
        return img
    rgb = img[..., :3]
    non_white = np.any(rgb < 0.985, axis=2)
    if not np.any(non_white):
        return img

    rows = np.where(np.any(non_white, axis=1))[0]
    cols = np.where(np.any(non_white, axis=0))[0]
    r0 = max(int(rows[0]) - pad, 0)
    r1 = min(int(rows[-1]) + pad + 1, img.shape[0])
    c0 = max(int(cols[0]) - pad, 0)
    c1 = min(int(cols[-1]) + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def _z_vector_to_slice(z: np.ndarray, brain_mask: nib.Nifti1Image, z_coord: int) -> np.ndarray:
    """Project a masked Z vector back into volume space and return one MNI z-coordinate slice."""
    mask_data = brain_mask.get_fdata() > 0
    n_mask = int(np.sum(mask_data))
    if z.size != n_mask:
        raise ValueError(f"Z-map has {z.size} voxels but mask has {n_mask}")

    voxel = nib.affines.apply_affine(np.linalg.inv(brain_mask.affine), [[0, 0, z_coord]])[0]
    z_slice = int(round(voxel[2]))
    if not (0 <= z_slice < mask_data.shape[2]):
        raise ValueError(
            f"z_coord={z_coord} maps to slice index {z_slice}, outside valid range "
            f"0-{mask_data.shape[2] - 1}"
        )

    volume = np.full(mask_data.shape, np.nan, dtype=np.float32)
    volume[mask_data] = z.astype(np.float32, copy=False)
    return np.rot90(volume[:, :, z_slice])


def plot_combined_subsampling_sensitivity_zmaps(results_dir: Path, project_dir: Path,
                                                args: argparse.Namespace,
                                                timestamp: str) -> list[Path]:
    """Create one figure per N with S-GLM/MUM rows and seed/repetition columns.

    Each panel is first rendered by plot.plot_brain so the combined figure uses
    the same nilearn brain-template view as the per-seed sensitivity Z-maps.
    """
    if args.subsampling_raw_dir is None:
        raw_dir = project_dir / "experiments" / "subsampling_experiment_UKB_R100" / "results" / "raw"
    else:
        raw_dir = Path(args.subsampling_raw_dir)
    if not raw_dir.is_dir():
        print(f"Subsampling raw directory not found: {raw_dir}", file=sys.stderr)
        return []

    z_by_N_rep = _load_subsampling_zmaps_by_rep(raw_dir, args.combined_N_list, args.seeds)
    all_vectors = [
        z
        for by_rep in z_by_N_rep.values()
        for by_method in by_rep.values()
        for z in by_method.values()
    ]
    if not all_vectors:
        print("No matching subsampling Z-map arrays found for requested N/reps.", file=sys.stderr)
        return []

    expected_voxels = all_vectors[0].size
    brain_mask = _load_matching_brain_mask(project_dir, expected_voxels)
    finite_chunks = [np.abs(z[np.isfinite(z)]) for z in all_vectors if np.isfinite(z).any()]
    finite_abs = np.concatenate(finite_chunks) if finite_chunks else np.array([], dtype=float)
    z_vmax = float(np.nanpercentile(finite_abs, 99.0)) if finite_abs.size else 1.0
    z_vmax = max(z_vmax, 1.0)

    output_paths = []
    for N in sorted(z_by_N_rep):
        if args.combined_zmap_output is None:
            output_png = results_dir / f"subsampling_zmaps_N{N}_zslice{args.combined_z_slice}_{timestamp}.png"
        else:
            output_base = Path(args.combined_zmap_output)
            output_png = output_base.with_name(f"{output_base.stem}_N{N}{output_base.suffix}")
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_pdf = output_png.with_suffix(".pdf")

        panel_dir = output_png.parent / f"{output_png.stem}_panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        panel_images: dict[str, dict[int, Path]] = {"S-GLM": {}, "MUM": {}}
        frequency_images: dict[str, Path] = {}
        for method in ("S-GLM", "MUM"):
            for rep in args.seeds:
                z_map = z_by_N_rep.get(N, {}).get(rep, {}).get(method)
                if z_map is None:
                    continue
                safe_method = method.replace("-", "").replace(" ", "_")
                panel_path = panel_dir / f"{safe_method}_seed{rep}.png"
                plot_brain(
                    p=z_map,
                    brain_mask=brain_mask,
                    slice_idx=args.combined_z_slice,
                    threshold=0,
                    vmin=-z_vmax,
                    vmax=z_vmax,
                    output_filename=str(panel_path),
                    colorbar=False,
                )
                panel_images[method][rep] = panel_path

        if args.include_sensitivity_frequency:
            frequency_maps = _sensitivity_frequency_for_N(
                project_dir,
                N,
                args.models,
                args.seeds,
                args.inference_method,
                args.alpha,
            )
            for method, (freq_map, n_maps) in frequency_maps.items():
                if method not in frequency_images:
                    safe_method = method.replace("-", "").replace(" ", "_")
                    panel_path = panel_dir / f"{safe_method}_sensitivity_frequency.png"
                    plot_brain(
                        p=freq_map,
                        brain_mask=brain_mask,
                        slice_idx=args.combined_z_slice,
                        threshold=0,
                        vmin=0,
                        vmax=1,
                        output_filename=str(panel_path),
                        colorbar=False,
                        cmap=args.sensitivity_frequency_cmap,
                    )
                    frequency_images[method] = panel_path

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "semibold",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })
        n_cols = len(args.seeds) + (1 if frequency_images else 0)
        fig, axes = plt.subplots(
            2,
            n_cols,
            figsize=(3.05 * n_cols, 4.65),
            constrained_layout=False,
        )
        axes = np.atleast_2d(axes)
        for row, method in enumerate(("S-GLM", "MUM")):
            for col, rep in enumerate(args.seeds):
                ax = axes[row, col]
                panel_path = panel_images[method].get(rep)
                if panel_path is None:
                    ax.axis("off")
                    continue
                ax.imshow(_crop_white_border(plt.imread(panel_path)))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(f"seed={rep}", fontsize=9, pad=3)
            if frequency_images:
                ax = axes[row, len(args.seeds)]
                panel_path = frequency_images.get(method)
                if panel_path is None:
                    ax.axis("off")
                else:
                    ax.imshow(_crop_white_border(plt.imread(panel_path)))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_frame_on(False)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    if row == 0:
                        ax.set_title("Sensitivity\nfrequency", fontsize=9, pad=3)
            axes[row, 0].text(
                -0.04,
                0.5,
                method,
                transform=axes[row, 0].transAxes,
                ha="right",
                va="center",
                rotation=90,
                fontsize=11,
                fontweight="bold",
            )

        fig.subplots_adjust(
            left=0.035,
            right=0.955,
            bottom=0.035,
            top=0.93,
            wspace=0.002,
            hspace=0.015,
        )
        sm = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=-z_vmax, vmax=z_vmax),
            cmap="inferno",
        )
        sm.set_array([])
        z_colorbar_axes = axes[:, :len(args.seeds)].ravel().tolist() if frequency_images else axes.ravel().tolist()
        cbar = fig.colorbar(
            sm,
            ax=z_colorbar_axes,
            location="right",
            shrink=0.92,
            pad=0.004,
            fraction=0.018,
        )
        cbar.set_label("Z statistic", rotation=270, labelpad=12)
        cbar.outline.set_visible(False)
        if frequency_images:
            sm_freq = mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=0, vmax=1),
                cmap=args.sensitivity_frequency_cmap,
            )
            sm_freq.set_array([])
            cbar_freq = fig.colorbar(
                sm_freq,
                ax=axes[:, len(args.seeds)].ravel().tolist(),
                location="right",
                shrink=0.92,
                pad=0.030,
                fraction=0.030,
            )
            cbar_freq.set_label("Frequency", rotation=270, labelpad=12)
            cbar_freq.outline.set_visible(False)
        fig.savefig(output_png, dpi=300)
        fig.savefig(output_pdf)
        plt.close(fig)
        print("Saved Z-map figure:", output_png)
        print("Saved Z-map PDF:", output_pdf)
        output_paths.append(output_png)

    return output_paths


def plot_sensitivity_z_maps(results_dir: Path, project_dir: Path, args: argparse.Namespace,
                            timestamp: str) -> list[dict]:
    """Plot age-effect Z-maps and significant-voxel frequency maps across subsets."""
    z_by_model_seed = {}
    for model in args.models:
        for seed in args.seeds:
            z_file = _find_result_file(results_dir, "z_values", model, seed, args.inference_method)
            if not z_file:
                print(f"Missing Z-map for model={model}, seed={seed}; skipping plot.", file=sys.stderr)
                continue
            z = _load_first_array(z_file, preferred_key="z_stats")
            if z.size == 0:
                print(f"Empty Z-map in {z_file}; skipping plot.", file=sys.stderr)
                continue
            z_by_model_seed[(model, seed)] = (z, z_file)

    if not z_by_model_seed:
        print("No Z-map files found; skipping sensitivity Z-map plotting.", file=sys.stderr)
        return []

    expected_voxels = next(iter(z_by_model_seed.values()))[0].size
    brain_mask = _load_matching_brain_mask(project_dir, expected_voxels)
    z_threshold = float(norm.ppf(1.0 - args.alpha))

    if args.zmap_dir is None:
        zmap_dir = results_dir / f"zmap_plots_{timestamp}"
    else:
        zmap_dir = Path(args.zmap_dir)
    zmap_dir.mkdir(parents=True, exist_ok=True)

    if args.z_vmax is not None:
        z_vmax = float(args.z_vmax)
    else:
        all_abs_z = np.concatenate([
            np.abs(z[np.isfinite(z)]) for z, _ in z_by_model_seed.values()
        ])
        z_vmax = float(np.nanpercentile(all_abs_z, 99.0)) if all_abs_z.size else z_threshold
        z_vmax = max(z_vmax, z_threshold)

    rows = []
    for model in args.models:
        significant_masks = []
        for seed in args.seeds:
            entry = z_by_model_seed.get((model, seed))
            if entry is None:
                continue
            z, z_file = entry
            if z.size != expected_voxels:
                print(
                    f"Z-map length mismatch for model={model}, seed={seed}: "
                    f"{z.size} != {expected_voxels}; skipping plot.",
                    file=sys.stderr,
                )
                continue

            significant = np.isfinite(z) & (np.abs(z) >= z_threshold)
            significant_masks.append(significant.astype(float))
            output_file = zmap_dir / f"Zmap_age_{model}_seed{seed}_{args.inference_method}.png"
            plot_brain(
                p=z,
                brain_mask=brain_mask,
                threshold=z_threshold,
                vmin=-z_vmax,
                vmax=z_vmax,
                output_filename=str(output_file),
            )
            rows.append({
                "model": model,
                "seed": seed,
                "z_file": z_file,
                "zmap_plot": str(output_file),
                "z_threshold": z_threshold,
                "n_significant_voxels": int(np.sum(significant)),
                "significant_voxel_rate": float(np.mean(significant)),
            })

        if significant_masks:
            frequency = np.mean(np.vstack(significant_masks), axis=0)
            output_file = zmap_dir / f"Significant_voxel_frequency_age_{model}_{args.inference_method}.png"
            plot_brain(
                p=frequency,
                brain_mask=brain_mask,
                threshold=0,
                vmin=0,
                vmax=1,
                output_filename=str(output_file),
                cmap=args.sensitivity_frequency_cmap,
            )
            rows.append({
                "model": model,
                "seed": "frequency",
                "z_file": "",
                "zmap_plot": str(output_file),
                "z_threshold": z_threshold,
                "n_significant_voxels": int(np.sum(frequency > 0)),
                "significant_voxel_rate": float(np.mean(frequency > 0)),
            })

    return rows


def main() -> int:
    args = get_args()
    project_dir = Path(__file__).resolve().parent
    ukb_dir = project_dir / "results" / f"UKB_{args.UKB_subject}"
    results_dir = ukb_dir / "subset_sensitivity"
    results_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    if not args.skip_runs:
        for seed in args.seeds:
            for model in args.models:
                cmd = build_command(args, model, seed)
                print("\n" + "=" * 80, flush=True)
                print(f"Running model={model}, seed={seed}", flush=True)
                print(" ".join(cmd), flush=True)
                if args.dry_run:
                    continue
                completed = subprocess.run(cmd, cwd=project_dir)
                if completed.returncode != 0:
                    failures.append({"model": model, "seed": seed, "returncode": completed.returncode})
                    if args.stop_on_error:
                        print(f"Stopping after failure: {failures[-1]}", file=sys.stderr)
                        return completed.returncode
                else:
                    moved = archive_run_outputs(ukb_dir, results_dir, model, seed)
                    if moved:
                        print(f"Archived {len(moved)} files to {results_dir}", flush=True)
    else:
        print("Skipping run.py calls and using existing result files.", flush=True)

    if args.dry_run:
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows = [
        summarize_one(results_dir, model, seed, args.alpha, args.inference_method)
        for model in args.models
        for seed in args.seeds
    ]
    stability_rows = summarize_pairwise_stability(
        results_dir, args.models, args.seeds, args.alpha, args.inference_method
    )

    summary_csv = results_dir / f"subset_sensitivity_summary_{timestamp}.csv"
    stability_csv = results_dir / f"subset_stability_pairwise_{timestamp}.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(stability_csv, stability_rows)

    zmap_rows = []
    if args.plot_z_maps:
        zmap_rows = plot_sensitivity_z_maps(results_dir, project_dir, args, timestamp)
        zmap_csv = results_dir / f"subset_sensitivity_zmap_plots_{timestamp}.csv"
        write_csv(zmap_csv, zmap_rows)

    combined_zmaps = []
    if args.plot_combined_z_maps:
        combined_zmaps = plot_combined_subsampling_sensitivity_zmaps(
            results_dir,
            project_dir,
            args,
            timestamp,
        )

    print("\nSaved sensitivity summary:", summary_csv)
    print("Saved pairwise stability summary:", stability_csv)
    if zmap_rows:
        print("Saved sensitivity Z-map plot summary:", zmap_csv)
    if combined_zmaps:
        print("Saved subsampling Z-map figures:")
        for path in combined_zmaps:
            print("  ", path)
    if failures:
        failure_csv = results_dir / f"subset_run_failures_{timestamp}.csv"
        write_csv(failure_csv, failures)
        print("Some runs failed. Saved failure summary:", failure_csv, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
