"""Regression fitting for spatial brain-lesion models.

Provides two classes:
  * BrainRegression_full      – PyTorch-based L-BFGS optimisation
  * BrainRegression_Approximate – Closed-form / iterative NumPy solver
"""

import logging
import time

import numpy as np
import torch

from model import SpatialBrainLesionModel, MassUnivariateRegression
from util import (
    fit_multiplicative_log_glm,
    fit_MUM_log_glm,
    SpatialGLM_compute_mu_mean,
    SpatialGLM_compute_P_mean,
)

logger = logging.getLogger(__name__)


class _BaseBrainRegression:
    """Shared data-loading and bookkeeping utilities for regression backends."""

    DESIGN_SCALE = 50.0

    def __init__(self, dtype=torch.float64, device='cpu'):
        self.dtype = dtype
        self.device = device
        self._kwargs = {"dtype": self.dtype, "device": self.device}

    @staticmethod
    def _is_multigroup_data(data):
        """Return True when ``data`` uses the object-array group format."""
        first_value = next(iter(data.values()))
        return (
            hasattr(first_value, "ndim")
            and first_value.ndim == 0
            and hasattr(first_value, "item")
            and isinstance(first_value.item(), dict)
            and {"X_spatial", "Y", "Z"}.issubset(first_value.item())
        )

    @staticmethod
    def _append_intercept_np(array):
        """Append an intercept column to a NumPy design matrix."""
        return np.concatenate([array, np.ones((array.shape[0], 1))], axis=1)

    @staticmethod
    def _append_intercept_torch(tensor, **kwargs):
        """Append an intercept column to a Torch design tensor."""
        intercept = torch.ones((tensor.shape[0], 1), **kwargs)
        return torch.cat([tensor, intercept], dim=1)

    def _scaled_with_intercept(self, array):
        """Scale a NumPy design by ``50 / n_rows`` and append intercept."""
        array = np.asarray(array, dtype=np.float64)
        return self._append_intercept_np(array * self.DESIGN_SCALE / array.shape[0])

    def _load_torch_multigroup_data(self, data):
        """Load multi-group data as Torch tensors for full-model fitting."""
        self.group_names = list(data.keys())
        self.B, self.Z, self.Y = {}, {}, {}

        for group_name in self.group_names:
            group_data = data[group_name].item()
            B = torch.tensor(group_data["X_spatial"], **self._kwargs)
            Z = torch.tensor(group_data["Z"], **self._kwargs)
            Y = torch.tensor(group_data["Y"], **self._kwargs)
            self.B[group_name] = self._append_intercept_torch(B, **self._kwargs)
            self.Z[group_name] = self._append_intercept_torch(Z, **self._kwargs)
            self.Y[group_name] = Y

        self._set_torch_group_dimensions()

    def _set_torch_group_dimensions(self):
        """Populate and validate dimensions for Torch multi-group data."""
        self.n_subjects, self.n_covariates = {}, {}
        self.n_voxels, self.n_bases = {}, {}

        for group_name in self.group_names:
            self.n_subjects[group_name], self.n_covariates[group_name] = self.Z[
                group_name
            ].shape
            self.n_voxels[group_name], self.n_bases[group_name] = self.B[
                group_name
            ].shape

        self._validate_common_group_dimensions()
        first_group = self.group_names[0]
        self.n_voxels_scalar = self.n_voxels[first_group]
        self.n_bases_scalar = self.n_bases[first_group]
        self.n_covariates_scalar = self.n_covariates[first_group]

    def _validate_common_group_dimensions(self):
        """Ensure all groups have common voxel, basis, and covariate counts."""
        if len(self.group_names) <= 1:
            return

        checks = {
            "voxels": self.n_voxels,
            "bases": self.n_bases,
            "covariates": self.n_covariates,
        }
        for label, values in checks.items():
            unique_values = {values[group_name] for group_name in self.group_names}
            if len(unique_values) > 1:
                raise ValueError(f"Groups have different number of {label}: {values}")

    def _load_numpy_data(self, data):
        """Load flat or multi-group data as NumPy arrays for approximate fitting."""
        if self._is_multigroup_data(data):
            self._load_numpy_multigroup_data(data)
        else:
            self._load_numpy_flat_data(data)

    def _load_numpy_multigroup_data(self, data):
        """Load grouped simulation data and concatenate subjects for shared fits."""
        self.group_names = list(data.keys())
        self.n_group = len(self.group_names)
        self.Y_dict, self.Z_dict, self.n_subject = {}, {}, {}

        first_group = self.group_names[0]
        self.B = self._scaled_with_intercept(data[first_group].item()["X_spatial"])
        self._N, self._P = self.B.shape

        for group_name in self.group_names:
            group_data = data[group_name].item()
            Y_g = group_data["Y"]
            Z_g = self._scaled_with_intercept(group_data["Z"])
            self.Y_dict[group_name] = Y_g
            self.Z_dict[group_name] = Z_g
            self.n_subject[group_name] = Z_g.shape[0]

        self.Z = np.concatenate([self.Z_dict[group] for group in self.group_names], axis=0)
        self.Y = np.concatenate([self.Y_dict[group] for group in self.group_names], axis=0)
        self._M, self._R = self.Z.shape

    def _load_numpy_flat_data(self, data):
        """Load legacy flat data into NumPy arrays."""
        self.group_names = None
        self.n_group = 1
        self.B = self._scaled_with_intercept(data["X_spatial"])
        self.Z = self._scaled_with_intercept(data["Z"])
        self.Y = data["Y"]
        self._M, self._R = self.Z.shape
        self._N, self._P = self.B.shape

    @staticmethod
    def _spatial_fit_kwargs(tol, max_iter, alpha, gradient_mode,
                            preconditioner_mode, nll_mode, block_size,
                            compute_nll):
        """Collect keyword arguments for S-GLM fitting."""
        return dict(
            tol=tol,
            max_iter=max_iter,
            alpha=alpha,
            gradient_mode=gradient_mode,
            preconditioner_mode=preconditioner_mode,
            nll_mode=nll_mode,
            block_size=block_size,
            compute_nll=compute_nll,
        )

    @staticmethod
    def _compute_spatial_gof(Z, B, beta, mode, block_size):
        """Compute SpatialBrainLesion goodness-of-fit quantities."""
        mu_mean, mu_std = SpatialGLM_compute_mu_mean(Z, B, beta, mode=mode, block_size=block_size)
        p_mean = SpatialGLM_compute_P_mean(Z, B, beta, mode=mode, block_size=block_size)
        return mu_mean, mu_std, p_mean

    @staticmethod
    def _compute_mum_p_mean(Z, beta):
        """Compute MassUnivariateRegression fitted probability summary."""
        mu = np.exp(Z @ beta)
        return np.mean(mu * np.exp(-mu), axis=0)


class BrainRegression_full(_BaseBrainRegression):
    """Full-data regression via PyTorch L-BFGS optimisation."""

    def __init__(self, dtype=torch.float64, device='cpu'):
        """Initialise with computation dtype and device."""
        super().__init__(dtype=dtype, device=device)

    def load_data(self, data):
        """Load and prepare data tensors (Y, B with intercept, Z with intercept)."""
        self._load_torch_multigroup_data(data)

    def init_model(self, model_name, **kwargs):
        """Instantiate the specified model with the given keyword arguments."""
        if model_name == "SpatialBrainLesion":
            self.model = SpatialBrainLesionModel(
                n_covariates=self.n_covariates_scalar,
                n_auxiliary=kwargs["n_auxiliary"],
                std_auxiliary=kwargs["std_auxiliary"],
                n_samples=kwargs["n_samples"],
                regression_terms=kwargs["regression_terms"],
                link_func=kwargs["link_func"],
                marginal_dist=kwargs["marginal_dist"],
                n_bases=self.n_bases_scalar,
                group_names=self.group_names,
                device=self.device,
                dtype=self.dtype,
            )
        elif model_name == "MassUnivariateRegression":
            self.model = MassUnivariateRegression(
                n_covariates=self.n_covariates_scalar,
                n_auxiliary=kwargs["n_auxiliary"],
                std_auxiliary=kwargs["std_auxiliary"],
                n_samples=kwargs["n_samples"],
                regression_terms=kwargs["regression_terms"],
                link_func=kwargs["link_func"],
                marginal_dist=kwargs["marginal_dist"],
                firth_penalty=kwargs["firth_penalty"],
                n_voxels=self.n_voxels_scalar,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Model {model_name} not implemented")
    
    def optimize_model(self, lr, iter, tolerance_change, tolerance_grad=1e-7,
                       history_size=100, line_search_fn="strong_wolfe"):
        """Run L-BFGS optimisation on the loaded model and data."""
        start_time = time.time()
        self.iteration = 0
        optimizer = torch.optim.LBFGS(
            params=self.model.parameters(),
            lr=lr,
            max_iter=iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )

        def closure():
            optimizer.zero_grad()
            preds = self.model(self.B, self.Y, self.Z)
            loss = self.model.get_loss(preds, self.Y, self.Z)
            logger.info("Iteration %d: Loss: %.6f", self.iteration, loss.item())
            self.iteration += 1
            loss.backward()
            return loss
        optimizer.step(closure)

        logger.info("Optimisation took %.1f s", time.time() - start_time)

    
class BrainRegression_Approximate(_BaseBrainRegression):
    """Approximate regression using closed-form / iterative NumPy solvers."""

    def __init__(self, simulated_dset, dtype=torch.float64, device='cpu'):
        """Initialise with dataset flag, dtype, and device."""
        super().__init__(dtype=dtype, device=device)
        self.simulated_dset = simulated_dset

    def load_data(self, data, model):
        """Load and prepare data tensors (Y, B with intercept, Z with intercept).

        Supports two formats:
          - Multi-group dict: keys are group names, each data[group].item()
            contains {"X_spatial", "Y", "Z"}.
          - Legacy flat: data contains "X_spatial", "Y", "Z" directly.
        """
        self._load_numpy_data(data)

    def run_regression(self, 
                       model: str, 
                       marginal_dist: str,
                       link_func: str,
                       tol: float = 1e-10,
                       max_iter: int = 1000, 
                       alpha: float = 1.0,
                       gradient_mode: str = "dask", 
                       preconditioner_mode: str = "approximate", 
                       nll_mode: str = "dask",
                       block_size: int = 10000, 
                       compute_nll: bool = False):
        """Fit the regression model and return estimated coefficients.

        For multi-group SpatialBrainLesion, fits each group independently
        and returns a dict {group_name: beta_g}.
        For MassUnivariateRegression, fits on concatenated data (shared beta).
        """
        start = time.time()
        fit_kwargs = self._spatial_fit_kwargs(
            tol, max_iter, alpha, gradient_mode,
            preconditioner_mode, nll_mode, block_size, compute_nll,
        )
        if model == "SpatialBrainLesion":
            if self.group_names and self.n_group > 1:
                beta = {}
                for group_name in self.group_names:
                    logger.info("Fitting SpatialBrainLesion for group %s", group_name)
                    beta[group_name] = fit_multiplicative_log_glm(
                        self.Z_dict[group_name],
                        self.B,
                        self.Y_dict[group_name],
                        **fit_kwargs,
                    )
            else:
                beta = fit_multiplicative_log_glm(
                    self.Z,
                    self.B,
                    self.Y,
                    **fit_kwargs,
                )
        elif model == "MassUnivariateRegression":
            beta = fit_MUM_log_glm(
                self.Z,
                self.B,
                self.Y,
                marginal_dist,
                link_func,
                tol=tol,
                max_iter=max_iter,
                alpha=alpha,
                nll_mode=nll_mode,
                block_size=block_size,
                compute_nll=compute_nll,
            )
        else:
            raise ValueError(f"Model {model} not implemented")
        logger.info("Regression completed in %.1f s", time.time() - start)
        return beta

    def goodness_of_fit(self, beta, model, mode="dask", block_size=100):
        """Compute goodness-of-fit statistics (mean/std of MU, mean of P).

        For multi-group SpatialBrainLesion with per-group beta (dict),
        returns dicts keyed by group name.
        """
        if model == "SpatialBrainLesion":
            if isinstance(beta, dict):
                MU_mean, MU_std, P_mean = {}, {}, {}
                for group_name in self.group_names:
                    mu_m, mu_s, p_m = self._compute_spatial_gof(
                        self.Z_dict[group_name],
                        self.B,
                        beta[group_name],
                        mode,
                        block_size,
                    )
                    MU_mean[group_name] = mu_m
                    MU_std[group_name] = mu_s
                    P_mean[group_name] = p_m
                return MU_mean, MU_std, P_mean
            return self._compute_spatial_gof(self.Z, self.B, beta, mode, block_size)
        if model == "MassUnivariateRegression":
            return None, None, self._compute_mum_p_mean(self.Z, beta)
        raise ValueError(f"Model {model} not implemented")