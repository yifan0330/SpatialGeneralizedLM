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

import numpy as np
import nibabel as nib
from scipy.stats import norm

from plot import plot_brain


DEFAULT_MODELS = ["SpatialBrainLesion", "MassUnivariateRegression"]


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

    print("\nSaved sensitivity summary:", summary_csv)
    print("Saved pairwise stability summary:", stability_csv)
    if zmap_rows:
        print("Saved sensitivity Z-map plot summary:", zmap_csv)
    if failures:
        failure_csv = results_dir / f"subset_run_failures_{timestamp}.csv"
        write_csv(failure_csv, failures)
        print("Some runs failed. Saved failure summary:", failure_csv, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
