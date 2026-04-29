import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import os

n_subject = [1000, 2000] # [1000]
model = "MassUnivariateRegression" #"SpatialBrainLesion"
distribution = "Poisson" # "NB"
inference = "FI" # "sandwich"
# ── Load p-values across seeds ──────────────────────────────────────────
base_dir = f"/well/nichols/users/pra123/brain_lesion_project/experiment/inference/brain/GRF_{n_subject}/{model}_{distribution}_log"

sorted_pvals_list = []
for seed in range(100):
    fpath = os.path.join(
        base_dir,
        f"brain_Inference_{inference}_Simulation_full_model_linear_random_seed_{seed}.npz",
    )
    if not os.path.exists(fpath):
        print(f"Seed {seed}: file not found, skipping")
        continue
    p = np.load(fpath, allow_pickle=True)["p_vals"]  # (1, n_voxel) or (2, n_voxel)
    # Remove NaN / invalid per row
    if p.ndim == 1:
        p = p.reshape(1, -1)
    n_rows = p.shape[0]
    row_sorted = []
    for row in range(n_rows):
        p_row = p[row]
        valid = np.isfinite(p_row) & (p_row >= 0) & (p_row <= 1)
        p_row = p_row[valid]
        row_sorted.append(np.sort(p_row))
    sorted_pvals_list.append(row_sorted)
    print(f"Seed {seed}: {p.shape} p-values loaded")

n_seeds = len(sorted_pvals_list)
n_comparisons = len(sorted_pvals_list[0])  # 1 or 2
n_voxel = sorted_pvals_list[0][0].shape[0]
print(f"\nLoaded {n_seeds} seeds, {n_comparisons} comparison(s), {n_voxel} voxels each")

# Stack sorted p-values per comparison and average across seeds
mean_sorted_pvals_per_comp = []
for comp in range(n_comparisons):
    matrix = np.stack([sorted_pvals_list[s][comp] for s in range(n_seeds)])  # (n_seeds, n_voxel)
    mean_sorted_pvals_per_comp.append(np.mean(matrix, axis=0))  # (n_voxel,)

# ── Expected quantiles & Beta CI ────────────────────────────────────────
k = np.arange(1, n_voxel + 1)
expected = (k - 0.5) / n_voxel

alpha = 0.05
ci_lower = stats.beta.ppf(alpha / 2, k, n_voxel + 1 - k)
ci_upper = stats.beta.ppf(1 - alpha / 2, k, n_voxel + 1 - k)

# ── Convert to -log10 scale ─────────────────────────────────────────────
exp_log = -np.log10(expected)
ci_lower_log = -np.log10(ci_upper + 1e-300)
ci_upper_log = -np.log10(ci_lower + 1e-300)

# ── Plot styling ────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2.5,
    'ytick.minor.size': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'figure.dpi': 150,
})

# ── Create figure: 1 panel for single group, 2 panels for group comparison ──
title_list = ["Group 1 $-$ Group 2", "Group 2 $-$ Group 1"]

fig, axes = plt.subplots(1, n_comparisons, figsize=(4.5 * n_comparisons, 4.5), squeeze=False)
axes = axes.ravel()

for comp in range(n_comparisons):
    ax = axes[comp]
    mean_sorted_pvals = mean_sorted_pvals_per_comp[comp]
    obs_log = -np.log10(mean_sorted_pvals + 1e-300)
    rejection_rate = (mean_sorted_pvals < 0.05).sum() / n_voxel

    # 95% CI band
    ax.fill_between(exp_log, ci_lower_log, ci_upper_log, alpha=0.20, color="#B0B0B0",
                    label=r"95\% CI (Beta)", zorder=1, linewidth=0)

    # Identity line
    max_val = max(exp_log.max(), obs_log.max(), ci_upper_log.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], color="#333333", linewidth=0.9, linestyle="--",
            label=r"$y = x$", zorder=2)

    # Thin out points for clarity
    thin = max(1, n_voxel // 2000)
    idx = np.arange(0, n_voxel, thin)
    tail_idx = np.arange(max(0, n_voxel - 200), n_voxel)
    idx = np.union1d(idx, tail_idx)

    ax.scatter(exp_log[idx], obs_log[idx], s=6, alpha=0.55, color="#2166AC",
               edgecolors="none", label="Mean observed", zorder=3, rasterized=True)

    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")

    # Legend
    leg = ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#999999",
                    framealpha=0.95, borderpad=0.6, handletextpad=0.5, labelspacing=0.4)
    leg.get_frame().set_linewidth(0.6)

    ax.set_aspect("equal")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Annotation
    annotation = f"{rejection_rate:.2%} rejected at $\\alpha = 0.05$"
    ax.text(0.97, 0.03, annotation,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontstyle="italic", color="#444444")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title for multi-group case
    if n_comparisons > 1:
        ax.set_title(title_list[comp], fontsize=13)

fig.tight_layout(pad=0.4)
os.makedirs("figures", exist_ok=True)
fig.savefig(f"figures/PP_plot_{model}_{distribution}_{inference}_{n_subject}.pdf",
            dpi=300, bbox_inches="tight", transparent=False)
fig.savefig(f"figures/PP_plot_{model}_{distribution}_{inference}_{n_subject}.png",
            dpi=300, bbox_inches="tight", transparent=False)
print(f"Saved to figures/PP_plot_{model}_{distribution}_{inference}_{n_subject}.{{pdf,png}}")
plt.show()
