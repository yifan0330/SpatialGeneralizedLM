from typing import List
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import time

class SpatialBrainLesionModel(nn.Module):

    def __init__(self,
                 n_covariates: int,
                 n_auxiliary: int,
                 n_bases: int,
                 n_samples: int = 100,
                 std_params: float = 1.0,
                 std_auxiliary: float = 1.0,
                 link_func: str = "logit",
                 marginal_dist: str = "Bernoulli",
                 regression_terms: List[str] = ["multiplicative", "additive"],
                 device: str = "cpu",
                 dtype = torch.float32):
        """ Spatial brain lesion model

        Args:
            n_covariates: Number of covariates
            n_auxiliary: Number of auxiliary variables
            n_bases: Number of bases for spatial representations
            n_samples: Number of samples for Monte Carlo approximation
            std_params: Standard deviation of Gaussian parameters
            std_auxiliary: Standard deviation of Gaussian auxiliary variables
            link_func: Link function for intensity function, options: "logit", "log"
            marginal_dist: Marginal distribution at each spatial location, options: "Bernoulli", "Poisson"
            regression_terms: Regression terms, options: ["multiplicative", "additive"]

        X: Spatial design matrix of shape (n_voxel, n_bases)
        Y: Binary lesion mask of shape (n_subject, n_voxel)
        Z: Covariates matrix of shape (n_subject, n_covariates)
        A: Random auxiliary variables of shape (n_subject, n_auxiliary)
        """
        super().__init__()
        self.n_covariates = n_covariates
        self.n_auxiliary = n_auxiliary
        self.n_bases = n_bases
        self.n_samples = n_samples
        self.std_params = std_params
        self.std_auxiliary = std_auxiliary
        if link_func == "logit":
            self.inverse_link_func = nn.Sigmoid()
        elif link_func == "log":
            self.inverse_link_func = torch.exp
        elif link_func == "arctanh":
            self.inverse_link_func = lambda z: (nn.Tanh()(z) + 1.) / 2.
        else:
            raise ValueError(f"Link function {link_func} not implemented")
        self.marginal_dist = marginal_dist
        self.regression_terms = regression_terms
        self.device = torch.device(device)
        self.dtype = dtype
        self._kwargs = {"device": self.device, "dtype": self.dtype}
        # regression coefficients for covariates
        self.beta = nn.Parameter(torch.randn(n_bases, self.n_covariates, **self._kwargs) * self.std_params)


    def forward(self, X, Y, Z):
        """Compute predicted probabilities for each group.

        Parameters
        ----------
        X : dict[str, Tensor] or Tensor  – spatial design matrices
        Y : dict[str, Tensor] or Tensor  – observed outcomes
        Z : dict[str, Tensor] or Tensor  – subject covariates

        Returns
        -------
        dict[str, Tensor] or Tensor – predicted probabilities per group
        """
        if isinstance(Z, dict):
            P = {}
            for group_name in Z:
                P[group_name] = self.inverse_link_func(
                    Z[group_name] @ self.beta.T @ X[group_name].T
                )
            return P
        # Single-group backward compatibility
        return self.inverse_link_func(Z @ self.beta.T @ X.T)

    def get_loss(self, P, Y, Z):
        """Compute total NLL summed across all groups."""
        if isinstance(Y, dict):
            total_nll = 0.0
            for group_name in Y:
                total_nll += self._group_nll(P[group_name], Y[group_name])
            return total_nll
        return self._group_nll(P, Y)

    def _group_nll(self, P, Y):
        """Compute NLL for a single group."""
        if self.marginal_dist == "Bernoulli":
            nll = -(torch.log(P) * Y + torch.log(1 - P) * (1 - Y)).mean()
        elif self.marginal_dist == "Poisson":
            nll = -(Y * torch.log(P) - P).mean()
        else:
            raise ValueError(f"Marginal distribution {self.marginal_dist} not supported")
        return nll

    def _neg_log_likelihood(marginal_dist, link_func, regression_terms, 
                            X_spatial, Y, Z, beta, device="cpu"):
        if link_func == "logit":
            inverse_link_func = nn.Sigmoid()
        elif link_func == "log":
            inverse_link_func = torch.exp
        # Compute probability function
        P = inverse_link_func(Z @ beta.T @ X_spatial.T) # shape: (n_subject, n_voxel)
        # negative log-likelihood
        if marginal_dist == "Bernoulli":
            nll = -(torch.log(P) * Y + torch.log(1 - P) * (1 - Y)).sum()
        elif marginal_dist == "Poisson":
            nll = -(Y * torch.log(P) - P).sum()
        return nll

class MassUnivariateRegression(nn.Module):
    def __init__(self,
                 n_covariates: int,
                 n_auxiliary: int,
                 n_voxels: int,
                 n_samples: int = 100,
                 std_params: float = .1,
                 std_auxiliary: float = 1.0,
                 link_func: str = "logit",
                 marginal_dist: str = "Bernoulli",
                 firth_penalty: bool = False,
                 regression_terms: List[str] = ["multiplicative", "additive"],
                 device: str = "cpu",
                 dtype = torch.float32):
        super().__init__()
        self.n_covariates = n_covariates
        self.n_auxiliary = n_auxiliary
        self.n_voxels = n_voxels
        self.n_samples = n_samples
        self.std_params = std_params
        self.std_auxiliary = std_auxiliary
        if link_func == "logit":
            self.inverse_link_func = nn.Sigmoid()
        elif link_func == "log":
            self.inverse_link_func = torch.exp
        elif link_func == "arctanh":
            self.inverse_link_func = lambda z: (nn.Tanh()(z) + 1.) / 2.
        else:
            raise ValueError(f"Link function {link_func} not implemented")
        self.marginal_dist = marginal_dist
        self.firth_penalty = firth_penalty
        self.regression_terms = regression_terms
        self.device = torch.device(device)
        self.dtype = dtype
        self._kwargs = {"device": self.device, "dtype": self.dtype}
        # regression coefficients for covariates
        self.beta = nn.Parameter(torch.randn(n_voxels, self.n_covariates, **self._kwargs) * self.std_params)

    def forward(self, X, Y, Z):
        """Compute predicted probabilities for each group.

        Parameters
        ----------
        X : dict[str, Tensor] or Tensor  – spatial design matrices (unused but kept for API)
        Y : dict[str, Tensor] or Tensor  – observed outcomes
        Z : dict[str, Tensor] or Tensor  – subject covariates

        Returns
        -------
        dict[str, Tensor] or Tensor – predicted probabilities per group
        """
        if isinstance(Z, dict):
            self.X = X
            P = {}
            for group_name in Z:
                P[group_name] = self.inverse_link_func(
                    Z[group_name] @ self.beta.T
                )
            return P
        self.X = X
        self.n_subject = Z.shape[0]
        return self.inverse_link_func(Z @ self.beta.T)

    def get_loss(self, P, Y, Z, eps=1e-6):
        """Compute total NLL summed across all groups, with optional Firth penalty."""
        if isinstance(Y, dict):
            total_nll = 0.0
            for group_name in Y:
                total_nll += self._group_nll(P[group_name], Y[group_name])
            if self.firth_penalty:
                # Firth penalty uses all groups concatenated
                P_all = torch.cat([P[g] for g in Y], dim=0)
                Z_all = torch.cat([Z[g] for g in Z], dim=0)
                total_nll += self._firth_penalty(P_all, Z_all, eps)
            return total_nll
        nll = self._group_nll(P, Y)
        if self.firth_penalty:
            nll += self._firth_penalty(P, Z, eps)
        return nll

    def _group_nll(self, P, Y):
        """Compute NLL for a single group."""
        if self.marginal_dist == "Bernoulli":
            nll = -(torch.log(P) * Y + torch.log(1 - P) * (1 - Y)).mean()
        elif self.marginal_dist == "Poisson":
            nll = -(Y * torch.log(P) - P).mean()
        else:
            raise ValueError(f"Marginal distribution {self.marginal_dist} not supported")
        return nll

    def _firth_penalty(self, P, Z, eps=1e-6):
        """Compute the Firth (half-log-det FI) penalty across all subjects."""
        # Precompute eye for regularization
        eye_reg = torch.eye(self.n_covariates, device=self.device) * eps
        total_penalty = 0.0

        for voxel_idx in range(self.n_voxels):
            # Get probabilities for this voxel
            p_voxel = P[:, voxel_idx]  # (n_subjects,)

            # Fisher Information for logistic/Poisson regression at this voxel
            if self.marginal_dist == "Bernoulli":
                weights = p_voxel * (1 - p_voxel)
            elif self.marginal_dist == "Poisson":
                weights = p_voxel
            else:
                raise ValueError(f"Marginal distribution {self.marginal_dist} not supported")
            sqrt_weights = torch.sqrt(weights)

            # Weighted design matrix
            Z_weighted = Z * sqrt_weights[:, None]  # (n_subjects, n_covariates)
            # Fisher Information matrix (n_covariates, n_covariates)
            FI = torch.mm(Z_weighted.t(), Z_weighted)
            FI += eye_reg
            # Cholesky
            L = torch.linalg.cholesky(FI)  # FI = L L^T
            # logdet(FI) = 2 * sum(log(diag(L)))
            half_logdet = torch.log(torch.diagonal(L)).sum()
            total_penalty += half_logdet
            del L, FI, Z_weighted, p_voxel, sqrt_weights, weights

        return total_penalty

    def _neg_log_likelihood(marginal_dist, link_func, regression_terms, 
                            X_spatial, Y, Z, beta_param, beta_other, device="cpu"):
        # Replace the zero row in beta_other with beta_param using differentiable operations
        # beta_param has shape (n_voxels,) - the parameter we're differentiating w.r.t.
        # beta_other has shape (n_covariates, n_voxels) - other parameters set to 0
        
        # If beta_param is 1D, reshape it to 2D
        if beta_param.dim() == 1:
            beta_param = beta_param.unsqueeze(0)  # shape: (1, n_voxels)
        
        # Find which row is all zeros and create a boolean mask for row-wise replacement
        zero_row_mask = torch.all(beta_other == 0, dim=1, keepdim=True)  # shape: (n_covariates, 1)
        
        # Use torch.where to replace only zero rows with beta_param
        # Broadcasting: zero_row_mask (n_covariates, 1) broadcasts to (n_covariates, n_voxels)
        beta = torch.where(zero_row_mask, beta_param, beta_other)  # shape: (n_covariates, n_voxels)

        if link_func == "logit":
            inverse_link_func = nn.Sigmoid()
        elif link_func == "log":
            inverse_link_func = torch.exp
        # Compute probability function
        P = inverse_link_func(Z @ beta) # shape: (n_subject, n_voxel)
        # negative log-likelihood
        if marginal_dist == "Bernoulli":
            nll = -(torch.log(P) * Y + torch.log(1 - P) * (1 - Y)).mean()
        elif marginal_dist == "Poisson":
            nll = -(Y * torch.log(P) - P).mean()
        return nll