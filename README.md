# SpatialGeneralizedLM

This repository implements computationally efficient spatial statistical models for brain lesion map analysis, including Kronecker-structured spatial GLMs and robust sandwich-based inference. Compared with mass-univariate regression as a baseline model, it is designed to scale to large-scale datasets (e.g. UK Biobank) without requiring the full design matrix, whose dimensions are the product of the number of voxels and the number of subjects.

## Methods & Algorithms

### Models (`experiment/model.py`)

| Model | Description |
|-------|-------------|
| **Spatial Brain Lesion Model** | Kronecker-structured (multiplicative) spatial GLM: the mean function factorises as *μ = g⁻¹(Z β Bᵀ)* where *Z* = subject covariates, *B* = spatial B-spline bases, and *β* = coefficient matrix. Supports Poisson and Negative Binomial (NB) model with log links. |
| **Mass Univariate Regression** | Standard voxel-wise GLM *μ = g⁻¹(Z β)* with independent *β* per voxel. Optional **Firth penalisation** (Jeffrey's prior) via ½ log det I(β) added to the negative log-likelihood. |

### Spatial Basis Functions (`experiment/bspline.py`)

- **Tensor-product B-splines** — recursive evaluation in 1D/2D/3D, with brain-mask subsetting and removal of weakly-supported bases.
- **Random Fourier Features (RFF)** — *φ(x) = √(2/D) cos(ωᵀx + b)* approximating an RBF kernel.
- **Quasi-Monte Carlo Fourier Features** — Sobol-sequence-based frequency sampling for improved coverage.

### Regression / Estimation (`experiment/regression.py`, `experiment/util.py`)

| Algorithm | Description |
|-----------|-------------|
| **Preconditioned gradient descent** | For the Kronecker-structured Poisson log-GLM. Uses an approximate Fisher-information preconditioner that factors as *kron(ZᵀWZ, BᵀWB)⁻¹*. Warm-started from an additive model. |
| **L-BFGS (PyTorch)** | Full-model optimisation with Strong Wolfe line search. |
| **L-BFGS-B (SciPy)** | Mass-univariate model fitting with analytic gradient. |
| **IRLS (Fisher scoring)** | Classic iteratively re-weighted least squares for Poisson log-link GLM. |
| **Efficient Kronecker operations** | *kron(Z, B) v* and *kron(Z, B)ᵀ diag(d) kron(Z, B)* computed via `einsum` block structure — the full *(MN × RP)* design matrix is never formed. Dask support for out-of-core computation. |

### Inference / Hypothesis Testing (`experiment/inference.py`)

| Method | Description |
|--------|-------------|
| **Wald Z-test** | Single-contrast test: *z = Cβ̂ / √(C Σ Cᵀ)*. |
| **Chi-square test** | Multi-df generalised linear hypothesis: *χ² = (Cβ̂)ᵀ [C Σ Cᵀ]⁻¹ (Cβ̂)*. |
| **Fisher information** | Exact: autograd Hessian of the NLL (PyTorch). Approximate: efficient Kronecker-structured *XᵀWX* via `einsum`. |
| **Sandwich (robust) covariance** | Memory-efficient *Σ = A⁻¹ C A⁻¹* with Kronecker-structured bread and cluster / iid meat variants. Cholesky solve with pseudo-inverse fallback. |
| **Delta-method variance** | Propagates coefficient variance to the probability scale: *Var(P) = P² · Var(η)*. |
| **SVD-based robust pseudo-inverse** | Eigenvalue thresholding (median rule) + symmetrisation. |

Three inference classes handle different data regimes:  
`BrainInference_full` (exact autograd), `BrainInference_Approximate` (Kronecker-structured), `BrainInference_UKB` (UK Biobank scale).

### Data Simulation (`experiment/data_simulation.py`)

| Simulator | Description |
|-----------|-------------|
| **Gaussian Random Field (GRF)** | White noise → Gaussian smoothing → probit threshold to produce spatially-correlated binary lesion maps. |
| **Bernoulli with neighbourhood expansion** | Background + covariate intensity functions (homogeneous or Gaussian-bump) sampled per voxel with 1D/2D/3D neighbourhood offsets. |
| **Spatially homogeneous null** | Bernoulli(0.01) per voxel — no covariate effect. |
| **Subject-homogeneous null** | Each subject gets an all-0 or all-1 lesion map. |
| **UK Biobank loader** | Loads and resamples real lesion NIfTIs to 2 mm MNI space with sex/age/headsize/CVR covariates. |

### Evaluation & Plotting

- **Monte Carlo evaluation** (`experiment/evaluation.py`) — bias, variance, MSE, and probability of underestimation over 100 repeated simulations, stratified by voxel lesion frequency.
- **QQ plots** with Beta confidence intervals and FDR (Benjamini–Hochberg) control line.
- **Brain stat maps** rendered via nilearn, with NIfTI export.

## Project Structure

- `experiment/`: Experimental scripts and analysis code

## Key Files

- `experiment/run.py`: Main experiment runner script
- `experiment/model.py`: Statistical models (Spatial GLM, Mass Univariate)
- `experiment/regression.py`: Regression fitting (L-BFGS, preconditioned gradient descent)
- `experiment/inference.py`: Inference (Wald test, sandwich estimator, Fisher information)
- `experiment/bspline.py`: B-spline and Fourier feature basis construction
- `experiment/data_simulation.py`: Synthetic and UK Biobank data generation
- `experiment/util.py`: Kronecker operations, gradient/preconditioner computation, NLL evaluation

## Requirements

- Python ≥ 3.11

All dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

### Setup with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create a virtual environment and install all dependencies in one step:

```bash
uv sync
```

This reads `uv.lock` and installs the exact same package versions used during development.

## Usage

Run experiments using the main script:

```bash
python experiment/run.py --model="SpatialBrainLesion" --n_group=2 --run_inference=True
```

See `experiment/run.py` for full command-line options.

## Data

Large data files (*.npz, *.nii.gz) are excluded from version control. You'll need to generate or obtain the required datasets separately.

## Results

Results and figures are generated in the `results/` and `figures/` directories (excluded from git due to size).
