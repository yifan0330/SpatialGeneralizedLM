"""Torch model definitions for spatial brain-lesion regression.

The module exposes two public model classes used by the regression and inference
pipelines:

* ``SpatialBrainLesionModel``: spatial basis model with optional per-group betas.
* ``MassUnivariateRegression``: independent voxel-wise GLM coefficients.

Shared link-function and likelihood logic lives in private helpers/base classes to
keep model implementations concise and consistent.
"""

from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn


_DEFAULT_REGRESSION_TERMS = ("multiplicative", "additive")
_EPS = 1e-6


def _normalise_regression_terms(regression_terms: Optional[Sequence[str]]) -> List[str]:
    """Return regression terms as a fresh list to avoid mutable defaults."""
    if regression_terms is None:
        return list(_DEFAULT_REGRESSION_TERMS)
    return list(regression_terms)


def _build_inverse_link(link_func: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create the inverse-link function used to map linear predictors to means."""
    if link_func == "logit":
        return torch.sigmoid
    if link_func == "log":
        return torch.exp
    if link_func == "arctanh":
        return lambda z: (torch.tanh(z) + 1.0) / 2.0
    raise ValueError(f"Link function {link_func} not implemented")


def _safe_log(tensor: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Numerically stable logarithm."""
    return torch.log(torch.clamp(tensor, min=eps))


def _distribution_nll(
    P: torch.Tensor,
    Y: torch.Tensor,
    marginal_dist: str,
    *,
    reduction: str = "mean",
    eps: float = _EPS,
) -> torch.Tensor:
    """Compute negative log-likelihood for a supported marginal distribution."""
    if marginal_dist == "Bernoulli":
        P = torch.clamp(P, min=eps, max=1.0 - eps)
        nll = -(_safe_log(P, eps) * Y + _safe_log(1.0 - P, eps) * (1.0 - Y))
    elif marginal_dist == "Poisson":
        P = torch.clamp(P, min=eps)
        nll = -(Y * _safe_log(P, eps) - P)
    elif marginal_dist == "NB":
        P = torch.clamp(P, min=eps)
        r = torch.tensor(1.0, dtype=P.dtype, device=P.device)
        p = torch.clamp(r / (r + P), min=eps, max=1.0 - eps)
        nll = -(
            torch.lgamma(Y + r)
            - torch.lgamma(r)
            - torch.lgamma(Y + 1.0)
            + r * _safe_log(p, eps)
            + Y * _safe_log(1.0 - p, eps)
        )
    else:
        raise ValueError(f"Marginal distribution {marginal_dist} not supported")

    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"Reduction {reduction} not supported")


class _BaseBrainLesionTorchModel(nn.Module):
    """Shared bookkeeping, link-function, and likelihood behaviour."""

    def __init__(
        self,
        *,
        n_covariates: int,
        n_auxiliary: int,
        n_samples: int,
        std_params: float,
        std_auxiliary: float,
        link_func: str,
        marginal_dist: str,
        regression_terms: Optional[Sequence[str]],
        device: str,
        dtype,
    ):
        super().__init__()
        self.n_covariates = n_covariates
        self.n_auxiliary = n_auxiliary
        self.n_samples = n_samples
        self.std_params = std_params
        self.std_auxiliary = std_auxiliary
        self.link_func = link_func
        self.inverse_link_func = _build_inverse_link(link_func)
        self.marginal_dist = marginal_dist
        self.regression_terms = _normalise_regression_terms(regression_terms)
        self.device = torch.device(device)
        self.dtype = dtype
        self._kwargs = {"device": self.device, "dtype": self.dtype}

    def _apply_inverse_link(self, linear_predictor: torch.Tensor) -> torch.Tensor:
        """Apply the configured inverse-link function."""
        return self.inverse_link_func(linear_predictor)

    def _group_nll(self, P: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute mean negative log-likelihood for one group."""
        return _distribution_nll(P, Y, self.marginal_dist, reduction="mean")

    def _sum_group_nll(self, P, Y) -> torch.Tensor:
        """Compute total NLL for dict or tensor inputs."""
        if isinstance(Y, dict):
            return sum(self._group_nll(P[group_name], Y[group_name]) for group_name in Y)
        return self._group_nll(P, Y)


class SpatialBrainLesionModel(_BaseBrainLesionTorchModel):
    """Spatial brain lesion model with spatial-basis regression coefficients.

    Parameters
    ----------
    n_covariates:
        Number of subject-level covariates.
    n_auxiliary:
        Number of auxiliary variables retained for API compatibility.
    n_bases:
        Number of spatial basis functions.
    group_names:
        Optional group names. If more than one group is supplied, a separate beta
        matrix is fitted for each group. Otherwise a shared ``beta`` is used.
    """

    def __init__(
        self,
        n_covariates: int,
        n_auxiliary: int,
        n_bases: int,
        n_samples: int = 100,
        std_params: float = 1.0,
        std_auxiliary: float = 1.0,
        link_func: str = "logit",
        marginal_dist: str = "Bernoulli",
        regression_terms: Optional[Sequence[str]] = None,
        group_names: Optional[Sequence[str]] = None,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        super().__init__(
            n_covariates=n_covariates,
            n_auxiliary=n_auxiliary,
            n_samples=n_samples,
            std_params=std_params,
            std_auxiliary=std_auxiliary,
            link_func=link_func,
            marginal_dist=marginal_dist,
            regression_terms=regression_terms,
            device=device,
            dtype=dtype,
        )
        self.n_bases = n_bases
        self.group_names = list(group_names) if group_names is not None else None
        self._initialise_beta_parameters()

    def _initialise_beta_parameters(self) -> None:
        """Initialise shared or group-specific beta parameters."""
        if self.group_names is not None and len(self.group_names) > 1:
            self.betas = nn.ParameterDict(
                {
                    group_name: nn.Parameter(
                        torch.randn(self.n_bases, self.n_covariates, **self._kwargs)
                        * self.std_params
                    )
                    for group_name in self.group_names
                }
            )
            self.beta = None
            return

        self.beta = nn.Parameter(
            torch.randn(self.n_bases, self.n_covariates, **self._kwargs) * self.std_params
        )
        self.betas = None

    def _get_beta(self, group_name=None) -> torch.Tensor:
        """Return beta for a specific group, or the shared beta."""
        if self.betas is not None and group_name is not None:
            return self.betas[group_name]
        return self.beta

    def _linear_predictor(self, X: torch.Tensor, Z: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Compute ``Z @ beta.T @ X.T`` for one group."""
        return Z @ beta.T @ X.T

    def forward(self, X, Y, Z):
        """Compute predicted probabilities/means for each group or tensor input."""
        if isinstance(Z, dict):
            return {
                group_name: self._apply_inverse_link(
                    self._linear_predictor(X[group_name], Z[group_name], self._get_beta(group_name))
                )
                for group_name in Z
            }

        return self._apply_inverse_link(self._linear_predictor(X, Z, self.beta))

    def get_loss(self, P, Y, Z):
        """Compute total negative log-likelihood across all groups."""
        return self._sum_group_nll(P, Y)

    @staticmethod
    def _neg_log_likelihood(
        marginal_dist,
        link_func,
        regression_terms,
        X_spatial,
        Y,
        Z,
        beta,
        device="cpu",
    ):
        """Static NLL used by autograd-based inference code."""
        inverse_link_func = _build_inverse_link(link_func)
        P = inverse_link_func(Z @ beta.T @ X_spatial.T)
        return _distribution_nll(P, Y, marginal_dist, reduction="sum")


class MassUnivariateRegression(_BaseBrainLesionTorchModel):
    """Mass-univariate regression model with one coefficient vector per voxel."""

    def __init__(
        self,
        n_covariates: int,
        n_auxiliary: int,
        n_voxels: int,
        n_samples: int = 100,
        std_params: float = 0.1,
        std_auxiliary: float = 1.0,
        link_func: str = "logit",
        marginal_dist: str = "Bernoulli",
        firth_penalty: bool = False,
        regression_terms: Optional[Sequence[str]] = None,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        super().__init__(
            n_covariates=n_covariates,
            n_auxiliary=n_auxiliary,
            n_samples=n_samples,
            std_params=std_params,
            std_auxiliary=std_auxiliary,
            link_func=link_func,
            marginal_dist=marginal_dist,
            regression_terms=regression_terms,
            device=device,
            dtype=dtype,
        )
        self.n_voxels = n_voxels
        self.firth_penalty = firth_penalty
        self.beta = nn.Parameter(
            torch.randn(n_voxels, self.n_covariates, **self._kwargs) * self.std_params
        )

    def _linear_predictor(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute voxel-wise linear predictors ``Z @ beta.T``."""
        return Z @ self.beta.T

    def forward(self, X, Y, Z):
        """Compute predicted probabilities/means for dict or tensor inputs.

        ``X`` and ``Y`` are retained in the signature for compatibility with the
        full regression training loop.
        """
        self.X = X
        if isinstance(Z, dict):
            return {
                group_name: self._apply_inverse_link(Z[group_name] @ self.beta.T)
                for group_name in Z
            }

        self.n_subject = Z.shape[0]
        return self._apply_inverse_link(self._linear_predictor(Z))

    def get_loss(self, P, Y, Z, eps=1e-6):
        """Compute total NLL, optionally adding the Firth penalty."""
        nll = self._sum_group_nll(P, Y)
        if not self.firth_penalty:
            return nll

        if isinstance(Y, dict):
            P_all = torch.cat([P[group_name] for group_name in Y], dim=0)
            Z_all = torch.cat([Z[group_name] for group_name in Y], dim=0)
            return nll + self._firth_penalty(P_all, Z_all, eps)
        return nll + self._firth_penalty(P, Z, eps)

    def _firth_penalty(self, P, Z, eps=1e-6):
        """Compute the Firth half-log-determinant penalty for all voxels.

        The implementation is batched over voxels and avoids a Python loop over
        ``n_voxels``. It computes ``0.5 * logdet(Z.T @ W_v @ Z + eps I)`` for
        each voxel ``v`` and sums the result.
        """
        if self.marginal_dist == "Bernoulli":
            weights = P * (1.0 - P)
        elif self.marginal_dist == "Poisson":
            weights = P
        else:
            raise ValueError(f"Marginal distribution {self.marginal_dist} not supported")

        weights = torch.clamp(weights, min=eps)
        fisher_info = torch.einsum("nr,ns,nv->vrs", Z, Z, weights)
        eye = torch.eye(self.n_covariates, device=Z.device, dtype=Z.dtype)
        fisher_info = fisher_info + eps * eye.unsqueeze(0)
        sign, log_abs_det = torch.linalg.slogdet(fisher_info)
        if torch.any(sign <= 0):
            raise RuntimeError("Firth penalty Fisher information is not positive definite")
        return 0.5 * log_abs_det.sum()

    @staticmethod
    def _neg_log_likelihood(
        marginal_dist,
        link_func,
        regression_terms,
        X_spatial,
        Y,
        Z,
        beta_param,
        beta_other,
        device="cpu",
    ):
        """Static NLL used by autograd-based inference code.

        ``beta_param`` replaces the all-zero row in ``beta_other`` while keeping
        the operation differentiable for Hessian calculations.
        """
        if beta_param.dim() == 1:
            beta_param = beta_param.unsqueeze(0)

        zero_row_mask = torch.all(beta_other == 0, dim=1, keepdim=True)
        beta = torch.where(zero_row_mask, beta_param, beta_other)
        inverse_link_func = _build_inverse_link(link_func)
        P = inverse_link_func(Z @ beta)
        return _distribution_nll(P, Y, marginal_dist, reduction="mean")