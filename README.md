# Brain Lesion Project

This repository contains code and experiments for brain lesion analysis using spatial statistical models.

## Project Structure

- `experiment/`: Experimental scripts and analysis code

## Key Files

- `experiment/run.py`: Main experiment runner script
- `experiment/brain_regression.py`: Brain regression analysis
- `experiment/model.py`: Statistical models for brain lesion analysis

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
