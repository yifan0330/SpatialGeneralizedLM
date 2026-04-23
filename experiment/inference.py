import torch
import numpy as np
import scipy
import logging
import time
import gc
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt
from model import SpatialBrainLesionModel, MassUnivariateRegression
from util import compute_mu, efficient_kronT_diag_kron, robust_inverse, robust_inverse_generalised, eigenspectrum
from plot import plot_brain, save_nifti
from statsmodels.stats.multitest import fdrcorrection

logger = logging.getLogger(__name__)

class BrainInference_full(object):
    def __init__(self, model,space_dim, marginal_dist, link_func, regression_terms, random_seed, fewer_voxels=False,
                dtype=torch.float64, device='cpu'):
        self.model = model
        self.space_dim=space_dim
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.random_seed = random_seed
        self.fewer_voxels = fewer_voxels
        self.dtype = dtype
        self.device = device
        self._kwargs = {"device": self.device, "dtype": self.dtype}
    
    def load_params(self, data, params):
        """Load spatial bases, covariates, outcomes, and fitted parameters.

        Parameters
        ----------
        data : dict
            Keys are group names. Each ``data[group].item()`` is a dict
            with keys ``"X_spatial"``, ``"Y"``, ``"Z"``.
        params : NpzFile
            Must contain ``"beta"`` and ``"P"`` (0-d object array wrapping
            a dict keyed by group name).
        """
        self.group_names = list(data.keys())
        self.n_group = len(self.group_names)

        # X_spatial is shared across groups — use the first group's
        first_group = self.group_names[0]
        X_spatial = data[first_group].item()["X_spatial"]
        self.X_spatial_array = np.concatenate(
            [X_spatial, np.ones((X_spatial.shape[0], 1))], axis=1,
        )
        self.X_spatial = torch.tensor(self.X_spatial_array, **self._kwargs)

        # Per-group data
        self.Y = {}
        self.Z = {}
        self.n_subject = {}
        self.n_covariates = {}
        for group_name in self.group_names:
            group_data = data[group_name].item()
            Y_g = group_data["Y"]
            Z_g = group_data["Z"]
            intercept_col = np.ones((Z_g.shape[0], 1))
            Z_with_intercept = np.concatenate([Z_g, intercept_col], axis=1)
            self.Y[group_name] = torch.tensor(Y_g, **self._kwargs)
            self.Z[group_name] = torch.tensor(Z_with_intercept, **self._kwargs)
            self.n_subject[group_name] = Z_with_intercept.shape[0]
            self.n_covariates[group_name] = Z_with_intercept.shape[1]

        # P from params (0-d object array wrapping dict)
        P_raw = params["P"]
        P_dict = P_raw.item() if P_raw.ndim == 0 else P_raw
        if isinstance(P_dict, dict):
            self.P_mean = np.stack(
                [np.mean(P_dict[g], axis=0) for g in self.group_names], axis=0,
            )  # (n_group, n_voxel)
        else:
            self.P_mean = np.mean(P_dict, axis=0, keepdims=True)
        self.eta = np.log(self.P_mean)

        # beta and spatial dimensions
        beta_raw = params["beta"]
        beta_val = beta_raw.item() if beta_raw.ndim == 0 else beta_raw
        if isinstance(beta_val, dict):
            # Per-group betas: dict {group_name: (n_bases, n_covariates)}
            self.beta_dict = {g: torch.tensor(beta_val[g], **self._kwargs) for g in self.group_names}
            self.beta_array_dict = {g: beta_val[g] for g in self.group_names}
            # For backward compat, set self.beta / beta_array to first group's
            first = self.group_names[0]
            self.beta = self.beta_dict[first]
            self.beta_array = self.beta_array_dict[first]
        else:
            # Single shared beta (single-group or legacy)
            self.beta = torch.tensor(beta_val, **self._kwargs)
            self.beta_array = beta_val
            self.beta_dict = {g: self.beta for g in self.group_names}
            self.beta_array_dict = {g: self.beta_array for g in self.group_names}
        self.n_voxel, self.n_bases = self.X_spatial.shape

    def create_contrast(self, contrast_vector=None, contrast_name=None):
        """Build and normalise the contrast vector over groups."""
        self.contrast_name = contrast_name
        if contrast_vector is None:
            if self.n_group == 1:
                # Single group: trivial identity contrast (unused in MUM path)
                self.contrast_vector = np.eye(1)
            else:
                # Default: consecutive pairwise differences c_g - c_{g+1}.
                # For 2 groups this is [[1, -1]], which tests the group difference
                # rather than each group's absolute effect.
                c = np.zeros((self.n_group - 1, self.n_group))
                for k in range(self.n_group - 1):
                    c[k, k] = 1
                    c[k, k + 1] = -1
                self.contrast_vector = c
        else:
            self.contrast_vector = np.array(contrast_vector).reshape(1, -1)
        if self.contrast_vector.shape[1] != self.n_group:
            raise ValueError(
                f"Contrast vector shape {self.contrast_vector.shape} "
                f"doesn't match number of groups ({self.n_group})."
            )
        # standardization (row sum 1)
        self.contrast_vector = self.contrast_vector / np.sum(np.abs(self.contrast_vector), axis=1).reshape((-1, 1))

    def run_inference(self, method="FI", inference_filename=None, fig_filename=None, lesion_mask=None, alpha=0.05):
        z_threshold = scipy.stats.norm.ppf(1-alpha)
        if not os.path.exists(inference_filename):
            print(f"[INFERENCE] Computing fresh inference → {inference_filename}")
            p_vals, z_stats = self._glh_con_group(method)
            np.savez(inference_filename, p_vals=p_vals, z_stats=z_stats)
        else:
            print(f"[INFERENCE] LOADING CACHED inference from {inference_filename}")
            loaded = np.load(inference_filename)
            p_vals = loaded["p_vals"]
            z_stats = loaded["z_stats"]
        print(f"[INFERENCE] z_stats: min={z_stats.min():.4f}, max={z_stats.max():.4f}, "
              f"mean={z_stats.mean():.4f}, std={np.std(z_stats):.4f}")
        print(f"[INFERENCE] significant (two-sided, alpha=0.05): "
              f"{np.count_nonzero(2.0 * scipy.stats.norm.sf(np.abs(z_stats)) < 0.05)}/{z_stats.size}")
        logger.info("p_vals shape: %s", p_vals.shape)
        logger.info("Plotting inference results to %s", fig_filename)
        os.makedirs(os.path.dirname(fig_filename), exist_ok=True)
        print("threshold", z_threshold)
        if z_stats.ndim == 2 and z_stats.shape[0] > 1:
            for i in range(z_stats.shape[0]):
                suffix = f"_contrast_{i}"
                out_fname = fig_filename.replace(".png", f"{suffix}.png")
                plot_brain(p=z_stats[i], brain_mask=lesion_mask, threshold=z_threshold, output_filename=out_fname)
        else:
            plot_brain(p=z_stats.ravel(), brain_mask=lesion_mask, threshold=z_threshold, output_filename=fig_filename)

    def _glh_con_group(self, method, batch_size=20):
        """Dispatch to model-specific inference method."""
        if self.model == "SpatialBrainLesion":
            return self._glh_SpatialBrainLesion(method)
        elif self.model == "MassUnivariateRegression":
            return self._glh_MassUnivariate()
        else:
            raise ValueError(f"Model {self.model} not supported for inference.")

    def _glh_MassUnivariate(self):
        """Wald test for MassUnivariateRegression (per-voxel beta).

        For each voxel j, beta_j is (n_covariates,) and the per-voxel
        Fisher information is F_j = Z^T W_j Z where W_j = diag(mu_{ij}).
        Cov(beta_j) = F_j^{-1}.
        """
        # Reconstruct mu per group, stack Z and Y across groups
        Z_all, Y_all = [], []
        for group in self.group_names:
            Z_all.append(self.Z[group].cpu().numpy())
            Y_all.append(self.Y[group].cpu().numpy())
        Z_np = np.concatenate(Z_all, axis=0)  # (M_total, n_covariates)
        Y_np = np.concatenate(Y_all, axis=0)  # (M_total, n_voxels)

        if self.link_func == "log":
            MU = np.exp(Z_np @ self.beta_array.T)  # (M_total, n_voxels)
            # FI per voxel: F_j = Z^T diag(mu_j) Z, shape (n_voxels, R, R)
            FI = np.einsum('im,ij,ik->jmk', Z_np, MU, Z_np)
        elif self.link_func == "logit":
            linear = Z_np @ self.beta_array.T
            MU = 1.0 / (1.0 + np.exp(-linear))
            FI = np.einsum('im,ij,ik->jmk', Z_np, MU * (1.0 - MU), Z_np)
        else:
            raise ValueError(f"Link function {self.link_func} not supported.")

        Cov_beta = np.linalg.pinv(FI)  # (n_voxels, R, R)

        # Numerator: contrast @ beta_j for each voxel
        # beta_array: (n_voxels, n_covariates), contrast_vector: (n_contrast, n_group)
        # For homogeneity test, contrast_vector is eye(1) = [[1]], so this is beta[:, 0]
        # For general tests, we need contrast over covariates — use the non-intercept sum
        n_cov = self.beta_array.shape[1]

        if self.n_group == 1:
            # Homogeneity: test non-intercept covariates
            contrast_beta = np.sum(self.beta_array[:, :n_cov - 1], axis=1)  # (n_voxels,)
            # Variance: sum of Cov(beta_s, beta_s) for non-intercept s
            var_beta = np.zeros(self.n_voxel)
            for s in range(n_cov - 1):
                var_beta += Cov_beta[:, s, s]
                # Add cross-covariance terms
                for t in range(s + 1, n_cov - 1):
                    var_beta += 2.0 * Cov_beta[:, s, t]
        else:
            # Group comparison for MUM with shared beta.
            # Per-group predicted mean at voxel j: eta_g(j) = bar_Z_g @ beta_j
            # Contrast: sum_g c_g * eta_g(j) = (sum_g c_g * bar_Z_g) @ beta_j = delta_Z @ beta_j
            # Under null (same generative process): bar_Z_1 ≈ bar_Z_2, so delta_Z ≈ 0.
            bar_Z_per_group = []
            for group in self.group_names:
                bar_Z_per_group.append(self.Z[group].cpu().numpy().mean(axis=0))  # (n_cov,)
            bar_Z_stack = np.stack(bar_Z_per_group, axis=0)  # (n_group, n_cov)
            # delta_Z = contrast_vector @ bar_Z_stack → (n_contrast, n_cov)
            delta_Z = self.contrast_vector @ bar_Z_stack  # (n_contrast, n_cov)
            # Numerator: delta_Z @ beta_j for each voxel j
            # beta_array: (n_voxels, n_cov), delta_Z: (n_contrast, n_cov)
            contrast_beta = delta_Z @ self.beta_array.T  # (n_contrast, n_voxels)
            # Variance: delta_Z @ Cov(beta_j) @ delta_Z^T for each voxel
            # Cov_beta: (n_voxels, n_cov, n_cov)
            var_beta = np.einsum(
                'ck,jkl,cl->cj', delta_Z, Cov_beta, delta_Z
            )  # (n_contrast, n_voxels)

        contrast_std = np.sqrt(np.maximum(var_beta, 0.0))

        z_stats = contrast_beta / np.where(contrast_std > 0, contrast_std, np.inf)
        # Two-sided p-value: 2 * P(Z > |z|)
        p_vals = 2.0 * scipy.stats.norm.sf(np.abs(z_stats))  # shape: (n_contrast, n_voxels)
        logger.info(
            "MUM p-values: min=%.4g, max=%.4g, significant=%d, shape=%s",
            np.min(p_vals), np.max(p_vals),
            np.count_nonzero(p_vals < 0.05), p_vals.shape,
        )
        return p_vals, z_stats

    def _glh_SpatialBrainLesion(self, method):
        """Wald test for SpatialBrainLesion (shared spatial-bases beta).

        Single-group (spatial homogeneity): tests whether non-intercept
        covariates (e.g. age) have a non-zero voxel-wise spatial effect
        beyond the intercept-only model.

        Multi-group (group comparison): tests whether group-specific
        mean voxel-wise log-intensities are equivalent.
        """
        all_bar_Z = {}
        for group in self.group_names:
            all_bar_Z[group] = self.Z[group].mean(dim=0).cpu().numpy()  # (n_covariates,)

        # --- Numerator of Wald test ---
        if self.n_group == 1:
            group = self.group_names[0]
            n_cov = self.n_covariates[group]
            beta_g = self.beta_array_dict[group]
            contrast_eta = np.sum(
                [self.X_spatial_array @ beta_g[:, s] for s in range(n_cov - 1)],
                axis=0,
            ).reshape(1, -1)  # (1, n_voxel)
            logger.info(
                "Homogeneity test: numerator from %d non-intercept covariates", n_cov - 1
            )
        else:
            all_eta_per_cov = {}
            for group in self.group_names:
                n_cov = self.n_covariates[group]
                beta_g = self.beta_array_dict[group]
                eta_per_cov = []
                for s in range(n_cov):
                    eta_s = all_bar_Z[group][s] * (self.X_spatial_array @ beta_g[:, s])
                    eta_per_cov.append(eta_s)
                all_eta_per_cov[group] = eta_per_cov
            group_eta = np.stack([
                np.sum(all_eta_per_cov[g], axis=0) for g in self.group_names
            ], axis=0)  # (n_group, n_voxel)

            contrast_eta = self.contrast_vector @ group_eta  # (n_contrast, n_voxel)
            logger.info("Group comparison: contrast_eta shape %s", contrast_eta.shape)

        # --- Estimate covariance of beta ---
        if method == "FI":
            all_F_beta = self._Fisher_info()
            all_cov_beta = {}
            for group in self.group_names:
                F_beta = all_F_beta[group]  # shape (P, R, P, R) from autograd Hessian
                n_cov = self.n_covariates[group]
                P_dim = self.n_bases
                # Reorder to (R*P, R*P): F[p1,r1,p2,r2] -> full[r1*P+p1, r2*P+p2]
                F_full = F_beta.transpose(1, 0, 3, 2).reshape(n_cov * P_dim, n_cov * P_dim)
                all_cov_beta[group] = np.linalg.inv(F_full + 1e-6 * np.eye(n_cov * P_dim))
                del F_beta
            del all_F_beta
        elif method == "sandwich":
            all_cov_beta = {}
            for group in self.group_names:
                Z_np = self.Z[group].cpu().numpy()
                Y_np = self.Y[group].cpu().numpy()
                beta_g = self.beta_array_dict[group]
                mu_group = np.exp(Z_np @ beta_g.T @ self.X_spatial_array.T)  # (M, N)
                n_cov = self.n_covariates[group]
                P = self.n_bases
                start_time = time.time()
                if self.marginal_dist == "NB":
                    r_nb = 1.0  # dispersion, matching model.py
                    bread_w = r_nb * mu_group / (r_nb + mu_group)
                    score_r = r_nb * (Y_np - mu_group) / (r_nb + mu_group)
                elif self.marginal_dist == "Poisson":
                    bread_w = mu_group
                    score_r = Y_np - mu_group
                else:
                    raise ValueError(f"Sandwich not implemented for {self.marginal_dist}")
                cov_full = self.poisson_sandwich_kron(
                    Z_np, self.X_spatial_array, Y_np, mu_group, meat="cluster",
                    bread_weights=bread_w, score_residuals=score_r,
                )
                logger.info("Sandwich cov for group %s computed in %.1fs", group, time.time() - start_time)
                all_cov_beta[group] = cov_full  # full (R*P, R*P) covariance
                del Z_np, Y_np, mu_group
        logger.info("Variance of beta computed")

        # --- Variance of the test statistic ---
        # Uses the FULL covariance (including cross-covariate blocks):
        #   Var(bar_eta_{g,j}) = sum_s sum_t  bar_z_s * bar_z_t * x_j^T Cov_st x_j
        P_dim = self.n_bases
        if self.n_group == 1:
            group = self.group_names[0]
            n_cov = self.n_covariates[group]
            var_total = np.zeros(self.n_voxel)
            for s in range(n_cov - 1):
                for t in range(n_cov - 1):
                    Cov_st = all_cov_beta[group][s * P_dim:(s + 1) * P_dim,
                                                  t * P_dim:(t + 1) * P_dim]
                    var_total += np.einsum(
                        'ij,jk,ik->i', self.X_spatial_array,
                        Cov_st, self.X_spatial_array
                    )
            del all_cov_beta
            contrast_var_bar_eta = var_total.reshape(1, -1)
        else:
            all_var_bar_eta = {}
            for group in self.group_names:
                n_cov = self.n_covariates[group]
                bar_z = all_bar_Z[group]
                var_g = np.zeros(self.n_voxel)
                for s in range(n_cov):
                    for t in range(n_cov):
                        Cov_st = all_cov_beta[group][s * P_dim:(s + 1) * P_dim,
                                                      t * P_dim:(t + 1) * P_dim]
                        var_g += bar_z[s] * bar_z[t] * np.einsum(
                            'ij,jk,ik->i', self.X_spatial_array,
                            Cov_st, self.X_spatial_array
                        )
                    logger.info("Variance for cov %d in %s", s, group)
                all_var_bar_eta[group] = var_g
            del all_cov_beta
            a = np.stack([
                all_var_bar_eta[group].reshape(1, -1)
                for group in self.group_names
            ], axis=0).squeeze(1)  # (n_group, n_voxel)
            logger.info("Aggregated variance shape: %s", a.shape)
            contrast_var_bar_eta = self.contrast_vector ** 2 @ a

        contrast_std_bar_eta = np.sqrt(contrast_var_bar_eta)
        z_stats = contrast_eta / contrast_std_bar_eta
        if self.n_group == 1:
            z_stats = z_stats.copy()
        else:
            # concatenate z-stats and -z-stats for two-sided test
            z_stats = np.concatenate([z_stats, -z_stats], axis=0)  # (2*n_contrast, n_voxel)
        p_vals = scipy.stats.norm.sf(z_stats)  # one-sided p-value for positive effect
        logger.info(
            "SGLM p-values: min=%.4g, max=%.4g, significant=%d, shape=%s",
            np.min(p_vals), np.max(p_vals),
            np.count_nonzero(p_vals < 0.05), p_vals.shape,
        )
        print("SGLM p-values: min=%.4g, max=%.4g, significant=%d, shape=%s",
            np.min(p_vals), np.max(p_vals),
            np.count_nonzero(p_vals < 0.05), p_vals.shape,)
        return p_vals, z_stats
    
    def _Fisher_info(self):
        """Compute or load cached per-group Fisher information matrices."""
        n_subject_list = [self.n_subject[g] for g in self.group_names]
        Fisher_info_filename = (
            f"{os.getcwd()}/results/{self.space_dim}/GRF_{n_subject_list}/"
            f"{self.model}_{self.marginal_dist}_{self.link_func}/Fisher_info_{self.random_seed}.npz"
        )
        os.makedirs(os.path.dirname(Fisher_info_filename), exist_ok=True)
        if os.path.exists(Fisher_info_filename):
            loaded = np.load(Fisher_info_filename, allow_pickle=True)
            all_H = {group: loaded[group] for group in loaded.files}
        else:
            start_time = time.time()
            all_H = {}
            for group in self.group_names:
                if self.model == "SpatialBrainLesion":
                    beta_g = self.beta_dict[group]
                    nll = lambda beta, g=group: SpatialBrainLesionModel._neg_log_likelihood(
                        self.marginal_dist, self.link_func, self.regression_terms,
                        self.X_spatial, self.Y[g], self.Z[g], beta, self.device)
                    H = torch.autograd.functional.hessian(nll, beta_g, create_graph=False)
                elif self.model == "MassUnivariateRegression":
                    beta_age = self.beta[:, 2]
                    beta_other = self.beta.clone()
                    beta_other[:, 2] = 0.0
                    nll = lambda beta, g=group: MassUnivariateRegression._neg_log_likelihood(
                        self.marginal_dist, self.link_func, self.regression_terms,
                        self.X_spatial, self.Y[g], self.Z[g], beta, beta_other, self.device)
                    H = torch.autograd.functional.hessian(nll, beta_age, create_graph=False)
                all_H[group] = H.detach().cpu().numpy()
            logger.info("Fisher information computed in %.1fs", time.time() - start_time)
            np.savez(Fisher_info_filename, **all_H)
        return all_H

    def poisson_sandwich_kron(self, Z, B, y, mu, *, meat="cluster", ridge=0.0,
                               bread_weights=None, score_residuals=None):
        """Memory-efficient sandwich covariance for GLM with log link.

        Exploits X[i,j,:] = kron(Z[i,:], B[j,:]) without materialising
        the full design matrix.

        Parameters
        ----------
        Z : (M, R) subject covariates
        B : (N, P) spatial bases
        y : (M, N) observed outcomes
        mu : (M, N) fitted mean (must be > 0)
        meat : 'cluster' or 'iid'
        ridge : ridge penalty added to bread
        bread_weights : (M, N) or None
            Custom weights for the bread (Fisher info).  Defaults to mu
            (Poisson).  For NB: r*mu/(r+mu).
        score_residuals : (M, N) or None
            Custom score contributions for the meat.  Defaults to y - mu
            (Poisson).  For NB: r*(y-mu)/(r+mu).

        Returns
        -------
        cov : (R*P, R*P) sandwich covariance of vec(beta)
        """
        Z = np.asarray(Z, dtype=float)
        B = np.asarray(B, dtype=float)
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)

        M, R = Z.shape
        N, P = B.shape
        p = R * P

        if bread_weights is None:
            bread_weights = mu  # Poisson default
        if score_residuals is None:
            score_residuals = y - mu  # Poisson default

        r = score_residuals  # (M, N)

        # Bread: A (p x p)
        w_bread = np.einsum('ik,il,ij->klj', Z, Z, bread_weights)  # (R, R, N)
        A = np.zeros((p, p))
        for k in range(R):
            for k2 in range(k, R):
                block = B.T @ (B * w_bread[k, k2, :, None])  # (P, P)
                A[k * P:(k + 1) * P, k2 * P:(k2 + 1) * P] = block
                if k != k2:
                    A[k2 * P:(k2 + 1) * P, k * P:(k + 1) * P] = block

        if ridge > 0:
            A += ridge * np.eye(p)

        # Meat
        meat_kind = meat.lower()
        if meat_kind == "cluster":
            Bt_r = B.T @ r.T  # (P, M)
            U = np.zeros((p, M))
            for k in range(R):
                U[k * P:(k + 1) * P, :] = Bt_r * Z[:, k][None, :]
            C = U
            Bmeat = None
        elif meat_kind == "iid":
            w_meat = np.einsum('ik,il,ij->klj', Z, Z, r ** 2)  # (R, R, N)
            Bmeat = np.zeros((p, p))
            for k in range(R):
                for k2 in range(k, R):
                    block = B.T @ (B * w_meat[k, k2, :, None])
                    Bmeat[k * P:(k + 1) * P, k2 * P:(k2 + 1) * P] = block
                    if k != k2:
                        Bmeat[k2 * P:(k2 + 1) * P, k * P:(k + 1) * P] = block
            C = None
        else:
            raise ValueError("meat must be 'iid' or 'cluster'.")

        # Solve: cov = A^{-1} meat A^{-1}
        try:
            L, low = scipy.linalg.cho_factor(A)
            if meat_kind == "cluster":
                Y = scipy.linalg.cho_solve((L, low), C)  # (p, M)
                cov = Y @ Y.T
            else:
                D = scipy.linalg.cho_solve((L, low), Bmeat)
                cov = scipy.linalg.cho_solve((L, low), D.T).T
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed — falling back to pseudo-inverse")
            Ainv = np.linalg.pinv(A)
            if meat_kind == "cluster":
                Y = Ainv @ C
                cov = Y @ Y.T
            else:
                cov = Ainv @ Bmeat @ Ainv

        cov = 0.5 * (cov + cov.T)
        return cov
    
    def batch_compute_covariance(self, var_P, Z, X, P, cov_beta_w, batch_size=20):
        n_subject = Z.shape[0]
        split_indices = np.arange(0, n_subject, batch_size)
        for left_index in tqdm(split_indices, total=len(split_indices)):
            right_index = min(left_index + batch_size, n_subject)
            Z_i = Z[left_index:right_index]
            P_i = P[left_index:right_index]
            var_P_i = self.compute_covariance(Z_i, X, P_i, cov_beta_w)
            var_P[left_index:right_index] = var_P_i[:]
            var_P.flush()
            del Z_i, P_i, var_P_i
            gc.collect()

    def compute_covariance(self, Z, X, P, cov_beta_w):
        unstacked_cov_beta_w  = np.stack(np.split(cov_beta_w, self.n_bases, axis=-1))
        unstacked_cov_beta_w = np.stack(np.split(unstacked_cov_beta_w, self.n_bases, axis=-2)) # [_P, _P, _R, _R]
        
        cov_A = unstacked_cov_beta_w @ Z.T[None, None, :, :] 
        cov_A = np.sum(cov_A * Z.T[None, None, :, :], axis=-2)
        cov_A = np.moveaxis(cov_A, -1, 0) # shape: (n_batch, n_bases, n_bases)
        var_eta = np.einsum('np,mpq,nq->mn', X, cov_A, X) # shape: (n_batch, n_voxel)
        var_P = P**2*var_eta # shape: (n_batch, n_voxel)
        # cov_eta = X[None, :, :] @ cov_A @ X.T[None, :, :] # shape: (n_batch, n_voxel, n_voxel)
        # cov_P = cov_eta * P[:, :, None] * P[:, None, :] # shape: (n_batch, n_voxel, n_voxel)
        del unstacked_cov_beta_w, P, cov_A, var_eta,
        gc.collect()
        
        return var_P
    
    def plot_1d(self, p_vals, filename, significance_level=0.05):
        # slice list
        fig, axes = plt.subplots(1, 2, figsize=(23, 11))

        # Subplot 3
        M, N = p_vals.shape
        # theoretical p-values 
        th_p = np.arange(1/float(N),1+1/float(N),1/float(N)) # shape: (n_voxel,)
        th_p_log = -np.log10(th_p)
        # kth order statistics
        k_array = np.arange(start=1, stop=N+1, step=1)
        # empirical confidence interval (estimated from p-values)
        z_1, z_2 = scipy.stats.norm.ppf(significance_level), scipy.stats.norm.ppf(1-significance_level)
        # Add the Beta confidence interval
        CI_lower = scipy.stats.beta.ppf(significance_level/2, k_array, N - k_array + 1)
        CI_upper = scipy.stats.beta.ppf(1 - significance_level/2, k_array, N - k_array + 1)

        group_comparison = [[0, 1], [1, 0]]
        title_list = ["group_0 - group_1", "group_1 - group_0"]
        for i in range(M):
            # sort the order of p-values under -log10 scale
            sorted_p_vals = np.sort(p_vals[i, :]) # shape: (n_voxel,)
            significance_percentage = np.sum(sorted_p_vals < 0.05) / N
            print(significance_percentage)
            axes[i].fill_between(th_p_log, -np.log10(CI_lower), -np.log10(CI_upper), color='grey', alpha=0.5,
                    label=f'{int((1-significance_level)*100)}% Beta CI')
            axes[i].plot(th_p_log, np.repeat(-np.log10(0.05), N), color='y', linestyle='--', label='threshold at -log10(0.05)')
            axes[i].plot(th_p_log, -np.log10(th_p), color='orange', linestyle='--', label='y=x')
            axes[i].plot(th_p_log, -np.log10(significance_level * th_p), color='red', linestyle='-', label='FDR(BH) control')
            axes[i].scatter(th_p_log, -np.log10(sorted_p_vals), c='#1f77b4', s=4)
            axes[i].set_xlim([0, np.max(-np.log10(k_array/N))])
            axes[i].set_ylim([0, np.max(-np.log10(k_array/N))]) 
            axes[i].set_xlabel("Expected -log10(P)", fontsize=20)
            axes[i].set_ylabel("Observed -log10(P)", fontsize=20)
            axes[i].set_title(f"{title_list[i]}: {significance_percentage*100:.2f}% voxels rejected", fontsize=30)
            axes[i].legend()

        # Save the figure
        fig.savefig(filename)

class BrainInference_Approximate(object):
    def __init__(self, model, marginal_dist, link_func, regression_terms, 
                dtype=torch.float64, device='cpu'):
        self.model = model
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.dtype = dtype
        self.device = device
    
    def load_params(self, data, params):
        # Support both legacy flat format and multi-group object-array format.
        first_key = list(data.keys())[0]
        first_val = data[first_key]
        is_multigroup = (
            hasattr(first_val, "ndim")
            and first_val.ndim == 0
            and hasattr(first_val, "item")
            and isinstance(first_val.item(), dict)
            and "Y" in first_val.item()
            and "Z" in first_val.item()
            and "X_spatial" in first_val.item()
        )

        if is_multigroup:
            self.group_names = list(data.keys())
            self.n_group = len(self.group_names)
            self.n_subject = {}

            first_group = self.group_names[0]
            X_spatial = data[first_group].item()["X_spatial"]
            B_scaled = X_spatial * 50 / X_spatial.shape[0]
            self.B = np.concatenate([B_scaled, np.ones((B_scaled.shape[0], 1))], axis=1)

            Y_all, Z_all = [], []
            for group_name in self.group_names:
                group_data = data[group_name].item()
                Y_g = group_data["Y"]
                Z_g = group_data["Z"]
                self.n_subject[group_name] = Y_g.shape[0]
                Y_all.append(Y_g)
                Z_all.append(Z_g)

            Z_cat = np.concatenate(Z_all, axis=0)
            Z_scaled = Z_cat * 50 / Z_cat.shape[0]
            self.Z = np.concatenate([Z_scaled, np.ones((Z_scaled.shape[0], 1))], axis=1)
            self.Y = np.concatenate(Y_all, axis=0)
        else:
            self.group_names = ["Group_1"]
            self.n_group = 1
            self.n_subject = {"Group_1": data["Y"].shape[0]}
            B_raw = data["X_spatial"]
            B_scaled = B_raw * 50 / B_raw.shape[0]
            self.B = np.concatenate([B_scaled, np.ones((B_scaled.shape[0], 1))], axis=1)
            Z_raw = data["Z"]
            Z_scaled = Z_raw * 50 / Z_raw.shape[0]
            self.Z = np.concatenate([Z_scaled, np.ones((Z_scaled.shape[0], 1))], axis=1)
            self.Y = data["Y"]

        self._M, self._R = self.Z.shape
        self._N, self._P = self.B.shape
        # Load parameters and re-scale
        beta_raw = params["beta"]
        self.beta = beta_raw.item() if getattr(beta_raw, "ndim", 1) == 0 else beta_raw
        if isinstance(self.beta, dict):
            raise NotImplementedError(
                "BrainInference_Approximate does not support per-group beta dict. "
                "Use full-model inference or provide a shared beta array."
            )
        # self.MU = compute_mu(self.rescaled_Z, self.rescaled_B, self.beta, mode="dask", block_size=1000) # shape: (n_subject*n_voxel, 1)
        self.MU = compute_mu(self.Z, self.B, self.beta, mode="dask", block_size=1000) # shape: (n_subject*n_voxel, 1)
        self.Y_reshape = self.Y.reshape(-1, 1) # shape: (n_subject*n_voxel, 1)
    
    def create_contrast(self, contrast_vector=None, contrast_name=None, polynomial_order=None):
        self.contrast_vector = contrast_vector
        self.contrast_name = contrast_name
        # Preprocess the contrast vector
        if contrast_vector is None:
            # Default: test the first non-intercept covariate (e.g. age).
            # Z columns are [cov_0 (age), ..., cov_{R-2}, intercept].
            # Contrast [1, 0, ..., 0] tests only the first covariate.
            c = np.zeros((1, self._R))
            c[0, 0] = 1
            self.contrast_vector = c
        else:
            self.contrast_vector = np.array(contrast_vector).reshape(1, -1)
        # raise error if dimension of contrast vector doesn't match
        if self.contrast_vector.shape[1] != self._R:
            raise ValueError(
                f"""The shape of contrast vector: {str(self.contrast_vector)}
                doesn't match with number of covariates (_R={self._R})."""
            )
        # standardization (row sum 1)
        self.contrast_vector = self.contrast_vector / np.sum(np.abs(self.contrast_vector), axis=1).reshape((-1, 1))
        
    def run_inference(self, method="FI", inference_filename=None, fig_filename=None):
        # Generalised linear hypothesis testing
        p_vals, z_stats = self._glh_con_group(method)
        # Save results if filename given
        if inference_filename is not None:
            os.makedirs(os.path.dirname(inference_filename), exist_ok=True)
            np.savez(inference_filename, p_vals=p_vals, z_stats=z_stats)
        # Plot the estimated P, standard error of P, and p-values
        if fig_filename is not None:
            self.plot_1d(p_vals, fig_filename, 0.05)

    def _glh_con_group(self, method, use_dask=True, batch_size=20):
        # Compute the per-covariate spatial effect maps: beta_map[s, j] = B[j,:] @ beta_reshape[:,s]
        # This directly tests H0: contrast @ beta_map[:, j] = 0 at each voxel j,
        # which is the correct null for "does covariate s have a spatially-varying effect?".
        # We do NOT weight by bar_Z because the age covariate is standardized (mean=0),
        # which would make bar_Z-weighted numerators identically zero.
        beta_reshape = self.beta.reshape(self._P, self._R, order="F")
        # beta_map: (n_covariates, n_voxel)  =  (beta_reshape.T @ B.T)
        beta_map = beta_reshape.T @ self.B.T                           # (R, N)
        contrast_eta_covariates = self.contrast_vector @ beta_map      # (S, N)
        logger.info("contrast_eta range: %.4g .. %.4g", contrast_eta_covariates.min(), contrast_eta_covariates.max())
        # Estimate the covariance of beta, from either FI or sandwich estimator
        start_time = time.time()
        if method == "FI":
            F_beta = efficient_kronT_diag_kron(self.Z, self.B, self.MU, use_dask=use_dask, block_size=1e4) # shape: (R*P, R*P)
            cov_beta = [robust_inverse(F_beta[i*self._P:(i+1)*self._P, i*self._P:(i+1)*self._P]+1e-6*np.eye(self._P)) for i in range(self._R)]
            del F_beta
            logger.info("Fisher Information computed in %.1fs", time.time() - start_time)
        elif method == "sandwich":
            MU_matrix = self.MU.reshape(self._M, self._N)
            cov_beta_full = self.poisson_sandwich_kron(
                self.Z, self.B, self.Y, MU_matrix, meat="cluster", ridge=0.0
            )  # (R*P, R*P)
            logger.info("Sandwich estimator computed in %.1fs", time.time() - start_time)

        if method == "FI":
            # Var(c[s] * B_j @ beta_s) = c[s]^2 * B_j^T Cov(beta_s) B_j  (block-diagonal approx)
            var_eta = list()
            for s in range(self._R):
                var_eta_s = np.einsum('ij,jk,ik->i', self.B, cov_beta[s], self.B) # (n_voxel,)
                var_eta.append(var_eta_s)
            var_eta = np.stack(var_eta, axis=0) # (R, N)
            del cov_beta
            gc.collect()
            contrast_var_bar_eta = self.contrast_vector**2 @ var_eta # (S, N)
        else:
            # Full sandwich variance: Var(c @ beta_map_j) = (c ⊗ B_j)^T Cov(beta) (c ⊗ B_j)
            # where c is the contrast over covariates (no bar_Z weighting)
            CB = np.einsum('ij,kl->ikjl', self.contrast_vector, self.B)  # (S, N, R, P)
            CB_flat = CB.reshape(self.contrast_vector.shape[0], self._N, -1)  # (S, N, R*P)
            tmp = np.einsum('snk,kl->snl', CB_flat, cov_beta_full)           # (S, N, R*P)
            contrast_var_bar_eta = np.sum(tmp * CB_flat, axis=-1)             # (S, N)

        contrast_std_bar_eta = np.sqrt(np.maximum(contrast_var_bar_eta, 0.0)) # (S, N)
        # Two-sided Wald test
        z_stats = contrast_eta_covariates / np.where(contrast_std_bar_eta > 0, contrast_std_bar_eta, np.inf)
        p_vals = 2.0 * scipy.stats.norm.sf(np.abs(z_stats))  # two-sided p-value
        logger.info(
            "SGLM p-values: min=%.4g, max=%.4g, significant=%d/%d, shape=%s",
            np.min(p_vals), np.max(p_vals),
            np.count_nonzero(p_vals < 0.05), p_vals.size, p_vals.shape,
        )
        return p_vals, z_stats

    def poisson_sandwich_kron(self,
                                Z,
                                B,
                                y,
                                mu,
                                *,
                                meat="cluster",
                                ridge=0.0):
        """Memory-efficient sandwich covariance for Poisson log-link GLM.

        This mirrors the UKB implementation and avoids materializing the
        full Kronecker design matrix.
        """
        Z  = np.asarray(Z,  dtype=float)
        B  = np.asarray(B,  dtype=float)
        y  = np.asarray(y,  dtype=float)
        mu = np.asarray(mu, dtype=float)

        M, R = Z.shape
        N, P = B.shape
        p = R * P

        if y.shape != (M, N) or mu.shape != (M, N):
            raise ValueError("y and mu must have shape (M, N) matching Z and B.")
        if not np.isfinite(Z).all() or not np.isfinite(B).all() or not np.isfinite(y).all():
            raise ValueError("Inputs Z/B/y contain NaN or Inf.")

        # Stabilize fitted means to prevent NaN/Inf propagation.
        mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=0.0)
        mu = np.clip(mu, 1e-12, 1e12)

        r = np.nan_to_num(y - mu, nan=0.0, posinf=0.0, neginf=0.0)

        w_bread = np.einsum('ik,il,ij->klj', Z, Z, mu)
        A = np.zeros((p, p))
        for k in range(R):
            for k2 in range(k, R):
                block = B.T @ (B * w_bread[k, k2, :, None])
                A[k*P:(k+1)*P, k2*P:(k2+1)*P] = block
                if k != k2:
                    A[k2*P:(k2+1)*P, k*P:(k+1)*P] = block

        if ridge > 0:
            A += ridge * np.eye(p)

        # Ensure numeric validity/symmetry before factorization.
        A = np.nan_to_num(A, nan=0.0, posinf=1e12, neginf=-1e12)
        A = 0.5 * (A + A.T)

        meat_kind = meat.lower()
        if meat_kind == "cluster":
            Bt_r = B.T @ r.T
            U = np.zeros((p, M))
            for k in range(R):
                U[k*P:(k+1)*P, :] = Bt_r * Z[:, k][None, :]
            C = U
            Bmeat = None
        elif meat_kind == "iid":
            w_meat = np.einsum('ik,il,ij->klj', Z, Z, r**2)
            Bmeat = np.zeros((p, p))
            for k in range(R):
                for k2 in range(k, R):
                    block = B.T @ (B * w_meat[k, k2, :, None])
                    Bmeat[k*P:(k+1)*P, k2*P:(k2+1)*P] = block
                    if k != k2:
                        Bmeat[k2*P:(k2+1)*P, k*P:(k+1)*P] = block
            C = None
        else:
            raise ValueError("meat must be 'iid' or 'cluster'.")

        try:
            ridge_eps = max(ridge, 1e-8)
            L, low = scipy.linalg.cho_factor(A + ridge_eps * np.eye(p))
            if meat_kind == "cluster":
                Y = scipy.linalg.cho_solve((L, low), C)
                cov = Y @ Y.T
            else:
                D   = scipy.linalg.cho_solve((L, low), Bmeat)
                cov = scipy.linalg.cho_solve((L, low), D.T).T
        except (np.linalg.LinAlgError, ValueError):
            print("Cholesky failed (or non-finite matrix) — falling back to pseudo-inverse")
            A_safe = np.nan_to_num(A, nan=0.0, posinf=1e12, neginf=-1e12)
            Ainv = np.linalg.pinv(A_safe + 1e-6 * np.eye(p))
            if meat_kind == "cluster":
                Y = Ainv @ C
                cov = Y @ Y.T
            else:
                Bmeat_safe = np.nan_to_num(Bmeat, nan=0.0, posinf=1e12, neginf=-1e12)
                cov = Ainv @ Bmeat_safe @ Ainv

        cov = 0.5 * (cov + cov.T)
        return cov
    
    def bread_term(self, Z, B, P, use_dask=True, block_size=1000):
        XTWX = efficient_kronT_diag_kron(Z, B, P, use_dask=use_dask, block_size=block_size) # shape: (n_covariates*n_bases, n_covariates*n_bases)
        XTWX = XTWX.reshape(self._P, self._R, self._P, self._R, order="F")
        bread_term = [robust_inverse(XTWX[:,i,:,i]+1e-6*np.eye(self._P)) for i in range(self._R)]
        del XTWX
        gc.collect()

        return bread_term
    
    def meat_term(self, Z, B, P, Y, use_dask=True, block_size=1000):
        # meat term: sum_M [D_i^TV_i^{-1}(Y_i-P_i)]*[D_i^TV_i^{-1}(Y_i-P_i)]^T
        R = Y - P # shape: (n_subject*n_voxel, 1)
        R = R.reshape(self._M, self._N)
        # 2. Compute the weighted spatial sum for each subject
        L = np.dot(R, B)  # shape: (n_subject, n_bases)
        # 3. For each subject, compute v_i = kron(Z[i], L[i])
        #    This uses einsum to compute the outer product for each subject,
        #    resulting in shape (n_subject, n_covariates, n_bases) and then reshapes it.
        V = [Z[:, i][:, None] * L for i in range(self._R)]
        # 4. Compute the meat term by summing the outer products of v_i
        meat_term = [Vi.T @ Vi for Vi in V]
        del R, L, V
        gc.collect()

        return meat_term

    def plot_1d(self, p_vals, filename, significance_level=0.05):
        # slice list
        fig, axes = plt.subplots(1, 2, figsize=(23, 11))

        # Subplot 3
        M, N = p_vals.shape
        # theoretical p-values 
        th_p = np.arange(1/float(N),1+1/float(N),1/float(N)) # shape: (n_voxel,)
        th_p_log = -np.log10(th_p)
        # kth order statistics
        k_array = np.arange(start=1, stop=N+1, step=1)
        # empirical confidence interval (estimated from p-values)
        z_1, z_2 = scipy.stats.norm.ppf(significance_level), scipy.stats.norm.ppf(1-significance_level)
        # Add the Beta confidence interval
        CI_lower = scipy.stats.beta.ppf(significance_level/2, k_array, N - k_array + 1)
        CI_upper = scipy.stats.beta.ppf(1 - significance_level/2, k_array, N - k_array + 1)

        group_comparison = [[0, 1], [1, 0]]
        title_list = ["group_0 - group_1", "group_1 - group_0"]
        for i in range(M):
            # sort the order of p-values under -log10 scale
            sorted_p_vals = np.sort(p_vals[i, :]) # shape: (n_voxel,)
            significance_percentage = np.sum(sorted_p_vals < 0.05) / N
            print(significance_percentage)
            axes[i].fill_between(th_p_log, -np.log10(CI_lower), -np.log10(CI_upper), color='grey', alpha=0.5,
                    label=f'{int((1-significance_level)*100)}% Beta CI')
            axes[i].plot(th_p_log, np.repeat(-np.log10(0.05), N), color='y', linestyle='--', label='threshold at -log10(0.05)')
            axes[i].plot(th_p_log, -np.log10(th_p), color='orange', linestyle='--', label='y=x')
            axes[i].plot(th_p_log, -np.log10(significance_level * th_p), color='red', linestyle='-', label='FDR(BH) control')
            axes[i].scatter(th_p_log, -np.log10(sorted_p_vals), c='#1f77b4', s=4)
            axes[i].set_xlim([0, np.max(-np.log10(k_array/N))])
            axes[i].set_ylim([0, np.max(-np.log10(k_array/N))]) 
            axes[i].set_xlabel("Expected -log10(P)", fontsize=20)
            axes[i].set_ylabel("Observed -log10(P)", fontsize=20)
            axes[i].set_title(f"{title_list[i]}: {significance_percentage*100:.2f}% voxels rejected", fontsize=30)
            axes[i].legend()

        # Save the figure
        fig.savefig(filename)

class BrainInference_UKB(object):
    def __init__(self, model, marginal_dist, link_func, regression_terms, 
                dtype=torch.float64, device='cpu'):
        self.model = model
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.dtype = dtype
        self.device = device

    def load_params(self, data, params):
        # Load data
        B, Z = data["X_spatial"], data["Z"]
        B = B * 50 / B.shape[0]
        Z = Z * 50 / Z.shape[0]
        self.B = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)
        self.Y = data["Y"]
        self.Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
        self._M, self._R = self.Z.shape
        self._N, self._P = self.B.shape
        # Load parameters and re-scale
        self.beta = params["beta"]
        # MU
        if self.model == "SpatialBrainLesion":
            self.MU = compute_mu(self.Z, self.B, self.beta, mode="dask", block_size=5000) # shape: (n_subject*n_voxel, 1)
            self.MU = self.MU.reshape(self._M, self._N) # shape: (n_subject, n_voxel)
            P = self.MU * np.exp(-self.MU) # shape: (n_subject, n_voxel)
            P_mean = np.mean(P, axis=0) # shape: (n_voxel,)

    def create_contrast(self, contrast_vector=None, contrast_name=None, polynomial_order=1):
        self.contrast_vector = contrast_vector
        self.contrast_name = contrast_name
        # Preprocess the contrast vector
        if contrast_name == "age":
            if polynomial_order == 1:
                self.contrast_vector = np.array([0, 1, 0, 0, 0]).reshape(-1, self._R)
            else:
                self.contrast_vector = np.array([
                                            [0, 0, 1, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 1, 0, 0, 0]
                                        ]).reshape(-1, self._R)
        else:
            self.contrast_vector = (
                np.eye(self._R)
                if contrast_vector is None
                else np.array(contrast_vector).reshape(-1, self._R)
            )
        self._S = self.contrast_vector.shape[0]
        # standardization (row sum 1)
        self.contrast_vector = self.contrast_vector / np.sum(np.abs(self.contrast_vector), axis=1).reshape((-1, 1))

    def run_inference(self, alpha=0.05, method="FI", lesion_mask=None, XTWX_filename=None, Fisher_info_filename=None,
                      meat_term_filename=None, bread_term_filename=None, p_vals_filename=None, 
                      z_vals_filename=None,fig_filename=None):
        self.XTWX_filename = XTWX_filename
        self.Fisher_info_filename = Fisher_info_filename
        self.meat_term_filename = meat_term_filename
        self.bread_term_filename = bread_term_filename
        self.p_vals_filename = p_vals_filename
        self.z_vals_filename = z_vals_filename
        self.fig_dir = os.path.dirname(fig_filename)
        z_threshold = scipy.stats.norm.ppf(1-alpha)
        # Generalised linear hypothesis testing
        if os.path.exists(self.p_vals_filename) and os.path.exists(self.z_vals_filename):
            p_vals = np.load(self.p_vals_filename)["p_vals"]
            z_stats = np.load(self.z_vals_filename)["z_stats"]
            print("loaded p-values and z-stats from file.")
        else:
            if self.model == "SpatialBrainLesion":
                p_vals, z_stats = self.SpatialGLM_glh_con_group(method, lesion_mask, True, 1e4)
            elif self.model == "MassUnivariateRegression":
                p_vals, z_stats = self.MUM_glh_con_group(lesion_mask)
            else:
                raise ValueError(f"Model {self.model} not supported for inference.")
            np.savez(self.p_vals_filename, p_vals=p_vals)
            np.savez(self.z_vals_filename, z_stats=z_stats)
            print("saved p-values and z-stats to file.")
        # Plot the estimated P, standard error of P, and p-values
        self.histogram_z_stats(z_stats, fig_filename.replace(".png", "_z_stats_histogram.png"))
        save_nifti(p_vals.flatten(), lesion_mask, os.path.join(self.fig_dir, f"p_vals_{self.model}_{method}.nii.gz"))
        save_nifti(z_stats.flatten(), lesion_mask, os.path.join(self.fig_dir, f"z_stats_{self.model}_{method}.nii.gz"))
        plot_brain(p=z_stats, brain_mask=lesion_mask, threshold=z_threshold, output_filename=fig_filename)
        
        # # FDR correction
        # rejected, corr_p = fdrcor
        # rection(p_vals.flatten(), alpha=0.05, method='indep')
        # # Clip to avoid 0 or 1 which produce +/-inf.
        # eps = 1e-300  # safe tiny number to avoid exact 0
        # corr_p_clipped = np.clip(corr_p, eps, 1.0 - 1e-16)
        # # Convert two-sided corrected p to a *signed* z:
        # corr_z = scipy.stats.norm.isf(corr_p_clipped) * np.sign(z_stats.flatten())
        # plot_brain(p=corr_z, brain_mask=lesion_mask, threshold=z_threshold, vmax=None, output_filename=fig_filename.replace(".png", "_FDR.png"))
    
    def SpatialGLM_glh_con_group(self, method, lesion_mask, use_dask=True, block_size=1e6):
        # Estimate the variance of beta, from either FI or sandwich estimator
        # Compute the Fisher information matrix
        if method == "FI":
            if not os.path.exists(self.XTWX_filename):
                XTWX = efficient_kronT_diag_kron(self.Z, self.B, self.MU, use_dask=use_dask, block_size=block_size) # shape: (n_covariates*n_bases, n_covariates*n_bases)
                np.savez(self.XTWX_filename, XTWX=XTWX)
            else:
                XTWX = np.load(self.XTWX_filename)["XTWX"]

        CB = np.einsum('ij,kl->ikjl', self.contrast_vector, self.B) # shape: (_S, _N, _R, _P)
        CB_flat = CB.reshape(self._S, self._N, -1) # shape: (_S, _N, _R*_P)
        # (C \otimes B) \beta
        CB_beta = CB_flat @ self.beta  # shape: (_S, _N, 1)
        CB_beta = CB_beta.squeeze(-1) # shape: (_S, _N)
        # get the path of self.fig_filename
        plot_brain(p=CB_beta.flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename=os.path.join(self.fig_dir, "numerator_map_SGLM.png"))
        # shape: (_S, _N) 
        if method == "FI":
            cov_beta = np.linalg.pinv(XTWX) # shape: (_R*_P, _R*_P)
            tmp = np.einsum('snk,kl->snl', CB_flat, cov_beta)         # (S, N, K)
            contrast_var_eta = np.sum(tmp * CB_flat, axis=-1, keepdims=True)  # (S, N, 1)
            plot_brain(p=np.sqrt(contrast_var_eta).flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename=os.path.join(self.fig_dir, "denominator_map_SGLM_FI.png"))
            # del bread_term, meat_term, cov_beta
        elif method == "sandwich":
            cov_beta, diag = self.poisson_sandwich_kron(self.Z, self.B, self.Y, self.MU, meat="iid", ridge=0, return_diagnostics=True)
            print(np.min(np.diag(cov_beta)), np.mean(np.diag(cov_beta)), np.max(np.diag(cov_beta)), "cov_beta diag stats")
            # meat_term = self.meat_term(self.Z, self.B, self.MU, self.Y) 
            # bread_term = self.bread_term(self.Z, self.B, self.MU, self.Y)
            tmp = np.einsum('snk,kl->snl', CB_flat, cov_beta)         # (S, N, K)
            contrast_var_eta = np.sum(tmp * CB_flat, axis=-1, keepdims=True)  # (S, N, 1)
            plot_brain(p=np.sqrt(contrast_var_eta).flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename=os.path.join(self.fig_dir, "denominator_map_SGLM_sandwich.png"))
            # del bread_term, meat_term, cov_beta
        if self._S == 1:
            contrast_std_eta = np.sqrt(contrast_var_eta) # shape: (_N, 1)
            # contrast_std_eta = np.clip(contrast_std_eta, a_min=1e-6, a_max=None)
            # Conduct Wald test (Z test)
            z_stats = CB_beta.reshape(-1, 1) / contrast_std_eta.reshape(-1, 1) # shape: (_N, 1)
            print(np.min(z_stats), np.max(z_stats), "z stats range")
            # one-sided p-values
            p_vals = scipy.stats.norm.sf(z_stats) # shape: (_N, 1)
        else:
            chi_square_stats = np.empty(shape=(0,))
            for j in range(self._N):
                CB_j = CB_flat[:, j, :]  # shape: (_S, _R*_P)
                CB_beta_j = CB_beta[:, j].reshape(1, self._S) # shape: (1, _S)
                v_j = CB_j @ cov_beta @ CB_j.T # shape: (_S, _S)
                v_j_inv = np.linalg.pinv(v_j) # shape: (_S, _S)
                chi_square_j = CB_beta_j @ v_j_inv @ CB_beta_j.T
                chi_square_stats = np.concatenate((chi_square_stats, chi_square_j.reshape(1,)), axis=0)
            p_vals = 1 - scipy.stats.chi2.cdf(chi_square_stats, df=self._S)
            print(p_vals.shape, np.count_nonzero(p_vals < 0.05))
            p_vals = p_vals.reshape((1,-1))
            # convert p-values to z-stats (one-sided)
            print(p_vals.shape, np.count_nonzero(p_vals < 0.05))
            z_stats = scipy.stats.norm.isf(p_vals / 2)
            # save to nifti file
    
        return p_vals, z_stats

    def MUM_glh_con_group(self, lesion_mask):
        # Conduct Wald test (Z test)
        contrast_beta_covariates = self.contrast_vector @ self.beta # shape: (1, n_voxel)
        # Estimate the variance of beta, from either FI or sandwich estimator
        # check if there is only one non-zero contrast
        if np.count_nonzero(self.contrast_vector) == 1:
            nonzero_index = np.nonzero(self.contrast_vector)[1].item()
            if self.link_func == "log":
                MU = np.exp(self.Z @ self.beta) # shape: (n_subject, n_voxel)
                FI = np.einsum('im,ij,ik->jmk', self.Z, MU, self.Z)  # shape: (N, R, R)
                Cov_beta = np.linalg.pinv(FI) # shape: (N, R, R)
            elif self.link_func == "logit":
                MU = 1 / (1 + np.exp(-(self.Z @ self.beta))) # shape: (n_subject, n_voxel)
                FI = np.einsum('im,ij,ik->jmk', self.Z, MU * (1 - MU), self.Z)  # shape: (N, R, R)
                Cov_beta = np.linalg.pinv(FI) # shape: (N, R, R)
            else:
                raise ValueError(f"Link function {self.link_func} not supported.")
        else:
            raise NotImplementedError("FI method only implemented for single non-zero contrast in MUM.")
        var_beta = Cov_beta[:, nonzero_index, nonzero_index] # shape: (n_voxel,)
        # print(np.min(var_beta), np.mean(var_beta), np.max(var_beta), "variance of beta")
        # Compute the numerator of the Z test
        contrast_std_beta = np.sqrt(var_beta) # shape: (1, n_voxel)
        plot_brain(p=contrast_beta_covariates.flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename="numerator_map_MUM.png")
        plot_brain(p=contrast_std_beta.flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename="denominator_map_MUM.png")
        # Conduct Wald test (Z test)
        z_stats_eta = contrast_beta_covariates / contrast_std_beta
        z_stats = z_stats_eta.reshape(-1)
        print(np.min(z_stats), np.max(z_stats), "z stats range")
        # z_stats = np.concatenate([z_stats_eta, -z_stats_eta], axis=0) # shape: (2, n_voxel)
        p_vals = 2 * scipy.stats.norm.sf(abs(z_stats))
        print(p_vals.shape, z_stats.shape)
        print(np.min(p_vals), np.max(p_vals), np.count_nonzero(p_vals < 0.05), p_vals.shape)

        return p_vals, z_stats
    
    def meat_term(self, Z, B, MU, Y, batch_M=1000):
        if MU.shape != Y.shape:
            MU = MU.reshape(Y.shape) # shape: (_M, _N)
        if not os.path.exists(self.meat_term_filename):
            meat_term_1 = np.zeros((self._P * self._R, self._P * self._R)) # shape: (_P*_R, _P*_R)
            W = Y - MU
            BW = W.dot(B)    # shape (M, P)
            T = (Z[:, :, None] * BW[:, None, :]).reshape(self._M, self._P * self._R)  # shape (M, PR)
            meat_term = T.T.dot(T)   # shape (PR, PR)
            del W, BW, T
            gc.collect()
            np.savez(self.meat_term_filename, meat_term=meat_term)
        else:
            print("Loading precomputed meat term...")
            meat_term = np.load(self.meat_term_filename)["meat_term"]

        return meat_term
    
    def bread_term(self, Z, B, MU, Y, dtype=np.float64, chunk_rows=256, epsilon=1e-6):
        if MU.shape != Y.shape:
            MU = MU.reshape(Y.shape)
        if not os.path.exists(self.bread_term_filename):
            print("Computing bread term...")
            start_time = time.time()
            bread_term = np.zeros((self._P * self._R, self._P * self._R)) # shape: (_P*_R, _P*_R)

            for i in range(self._M): 
                print(f"Processing subject {i+1}/{self._M}", end='\r')
                # X_i = np.kron(Z[i,:], B) # shape: (_N, _P*_R) 
                # U_i = X_i.T * np.sqrt(MU[i, :]) # shape: (_P*_R, _N) 
                # bread_term += U_i @ U_i.T # shape: (_P*_R, _P*_R)
                zi = Z[i, :]                    # shape: (R,)
                mu_i = MU[i, :]          
                G_B = B.T @ (mu_i[:, None] * B)
                G_z = np.outer(zi, zi)          # (R, R)
                # Accumulate
                bread_term += np.kron(G_z, G_B)
            # print(np.min(np.diag(bread_term)), np.mean(np.diag(bread_term)), np.max(np.diag(bread_term)), "bread term diag stats")
            # bread_term += epsilon * np.eye(self._P * self._R)
            # print("Added epsilon {} to bread term".format(epsilon))
            print(np.min(np.diag(bread_term)), np.mean(np.diag(bread_term)), np.max(np.diag(bread_term)), "bread term diag stats")
            print("Time taken for bread term computation:", time.time() - start_time)
            del Z, B, MU, Y
            gc.collect()
            np.savez(self.bread_term_filename, bread_term=bread_term)
        else:
            print("Loading precomputed bread term...")
            bread_term = np.load(self.bread_term_filename)["bread_term"]

        return bread_term

    def poisson_sandwich_kron(self,
                                Z,                 # shape (M, R) - subject covariates
                                B,                 # shape (N, P) - spatial bases
                                y,                 # shape (M, N)
                                mu,                # shape (M, N)
                                *,
                                meat="cluster",
                                ridge=0.0,
                                return_diagnostics=False
                                ):
        """
        Memory-efficient sandwich covariance for Poisson log-link GLM,
        exploiting  X[i,j,:] = kron(Z[i,:], B[j,:])  (never materialised).

        The full design X would be  (M, N, R*P)  which is ~24 GB for
        typical problem sizes.  This function avoids forming it.

        Bread  A = sum_i X_i^T diag(mu_i) X_i
            Block (k,k') of A = B^T diag(w_{kk'}) B
            where  w_{kk'}[j] = sum_i Z[i,k] Z[i,k'] mu[i,j]

        Cluster meat  C_i = kron(z_i, B^T r_i)
        iid     meat  block structure same as bread but with r^2
        """
        Z  = np.asarray(Z,  dtype=float)
        B  = np.asarray(B,  dtype=float)
        y  = np.asarray(y,  dtype=float)
        mu = np.asarray(mu, dtype=float)

        M, R = Z.shape
        N, P = B.shape
        p = R * P

        if y.shape != (M, N) or mu.shape != (M, N):
            raise ValueError("y and mu must have shape (M, N) matching Z and B.")
        if np.any(mu <= 0):
            raise ValueError("All mu must be > 0.")

        r = y - mu                                          # (M, N)

        # ------------------------------------------------------------------
        # Bread:  A  (p x p)
        # w[k,l,j] = sum_i Z[i,k]*Z[i,l]*mu[i,j]
        # ------------------------------------------------------------------
        w_bread = np.einsum('ik,il,ij->klj', Z, Z, mu)     # (R, R, N)

        A = np.zeros((p, p))
        for k in range(R):
            for k2 in range(k, R):
                block = B.T @ (B * w_bread[k, k2, :, None]) # (P, P)
                A[k*P:(k+1)*P, k2*P:(k2+1)*P] = block
                if k != k2:
                    A[k2*P:(k2+1)*P, k*P:(k+1)*P] = block  # symmetric

        if ridge > 0:
            A += ridge * np.eye(p)

        # ------------------------------------------------------------------
        # Meat
        # ------------------------------------------------------------------
        meat_kind = meat.lower()

        if meat_kind == "cluster":
            # U_i = X_i^T r_i = kron(z_i, B^T r_i)
            Bt_r = B.T @ r.T                                # (P, M)
            U = np.zeros((p, M))
            for k in range(R):
                U[k*P:(k+1)*P, :] = Bt_r * Z[:, k][None, :] # (P, M)
            C = U                                            # (p, M)
            Bmeat = None

        elif meat_kind == "iid":
            # Same block structure as bread but weighted by r^2
            w_meat = np.einsum('ik,il,ij->klj', Z, Z, r**2)  # (R, R, N)
            Bmeat = np.zeros((p, p))
            for k in range(R):
                for k2 in range(k, R):
                    block = B.T @ (B * w_meat[k, k2, :, None])
                    Bmeat[k*P:(k+1)*P, k2*P:(k2+1)*P] = block
                    if k != k2:
                        Bmeat[k2*P:(k2+1)*P, k*P:(k+1)*P] = block
            C = None
        else:
            raise ValueError("meat must be 'iid' or 'cluster'.")

        # ------------------------------------------------------------------
        # Solve:  cov = A^{-1} meat A^{-1}
        # ------------------------------------------------------------------
        try:
            L, low = scipy.linalg.cho_factor(A)
            if meat_kind == "cluster":
                Y = scipy.linalg.cho_solve((L, low), C)           # A^{-1} U,  (p, M)
                cov = Y @ Y.T
            else:
                D   = scipy.linalg.cho_solve((L, low), Bmeat)     # A^{-1} Bmeat
                cov = scipy.linalg.cho_solve((L, low), D.T).T     # (A^{-1} Bmeat) A^{-1}
        except np.linalg.LinAlgError:
            print("Cholesky failed — falling back to pseudo-inverse")
            Ainv = np.linalg.pinv(A)
            if meat_kind == "cluster":
                Y = Ainv @ C
                cov = Y @ Y.T
            else:
                cov = Ainv @ Bmeat @ Ainv

        cov = 0.5 * (cov + cov.T)

        if return_diagnostics:
            diag_info = {
                "method": "kron_cholesky",
                "meat": meat_kind,
                "ridge": ridge,
                "M": M, "N": N, "R": R, "P": P, "p": p,
            }
            return cov, diag_info
        return cov

    def histogram_z_stats(self, z_stats, filename):
        plt.figure(figsize=(10, 6))
        plt.hist(z_stats.flatten(), bins=100, color='blue', alpha=0.7, edgecolor='black')
        plt.title('Histogram of Z-statistics', fontsize=16)
        plt.xlabel('Z-statistic', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(filename)
        plt.close()