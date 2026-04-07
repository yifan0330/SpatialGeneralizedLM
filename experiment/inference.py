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
        self.beta = torch.tensor(params["beta"], **self._kwargs)
        self.beta_array = params["beta"]
        self.n_voxel, self.n_bases = self.X_spatial.shape

    def create_contrast(self, contrast_vector=None, contrast_name=None):
        """Build and normalise the contrast vector over groups."""
        self.contrast_name = contrast_name
        self.contrast_vector = (
            np.eye(self.n_group)
            if contrast_vector is None
            else np.array(contrast_vector).reshape(1, -1)
        )
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
            p_vals, z_stats = self._glh_con_group(method)
            np.savez(inference_filename, p_vals=p_vals, z_stats=z_stats)
        else:
            loaded = np.load(inference_filename)
            p_vals = loaded["p_vals"]
            z_stats = loaded["z_stats"]
        logger.info("p_vals shape: %s", p_vals.shape)
        logger.info("Plotting inference results to %s", fig_filename)
        plot_brain(p=z_stats.ravel(), brain_mask=lesion_mask, threshold=z_threshold, output_filename=fig_filename)


    def _glh_con_group(self, method, batch_size=20):
        """Generalised linear hypothesis test (Wald test).

        Single-group (spatial homogeneity): tests whether non-intercept
        covariates (e.g. age) have a non-zero voxel-wise spatial effect
        beyond the intercept-only model.

        Multi-group (group comparison): tests whether group-specific
        mean voxel-wise log-intensities are equivalent.
        """
        all_bar_Z = {}
        for group in self.group_names:
            all_bar_Z[group] = self.Z[group].mean(dim=0).cpu().numpy()  # (n_covariates,)

        n_subject_list = list(self.n_subject.values())

        # --- Per-covariate contribution to mean linear predictor ---
        # bar_eta_{gs,j} = bar_Z_{gs} * X_j @ beta_s
        all_eta_per_cov = {}
        for group in self.group_names:
            n_cov = self.n_covariates[group]
            eta_per_cov = []
            for s in range(n_cov):
                eta_s = all_bar_Z[group][s] * (self.X_spatial_array @ self.beta_array[:, s])
                eta_per_cov.append(eta_s)  # (n_voxel,)
            all_eta_per_cov[group] = eta_per_cov

        # --- Numerator of Wald test ---
        if self.n_group == 1:
            # Spatial homogeneity: test whether non-intercept coefficients
            # (e.g. age) have a non-zero spatial effect.
            # H0: X @ beta_s = 0 for each non-intercept covariate s.
            # bar_Z is NOT used here — we test the coefficient itself,
            # not the linear predictor evaluated at the mean covariate.
            group = self.group_names[0]
            n_cov = self.n_covariates[group]
            # Each non-intercept covariate contributes X @ beta_s
            contrast_eta = np.sum(
                [self.X_spatial_array @ self.beta_array[:, s] for s in range(n_cov - 1)],
                axis=0,
            ).reshape(1, -1)  # (1, n_voxel)
            logger.info(
                "Homogeneity test: numerator from %d non-intercept covariates", n_cov - 1
            )
        else:
            # Group comparison: contrast across group-level mean linear predictors.
            # For a zero-sum contrast the shared intercept cancels automatically.
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
                F_beta = all_F_beta[group]
                cov_beta = [np.linalg.inv(F_beta[:, i, :, i] + 1e-6 * np.eye(self.n_bases))
                            for i in range(self.n_covariates[group])]
                all_cov_beta[group] = cov_beta
                del F_beta
            del all_F_beta
        elif method == "sandwich":
            all_cov_beta = {}
            for group in self.group_names:
                Z_np = self.Z[group].cpu().numpy()
                Y_np = self.Y[group].cpu().numpy()
                g_idx = self.group_names.index(group)
                # Reconstruct per-subject mu from eta; mu = exp(Z @ beta^T @ X^T)
                mu_group = np.exp(Z_np @ self.beta_array.T @ self.X_spatial_array.T)  # (M, N)
                n_cov = self.n_covariates[group]
                P = self.n_bases  # number of spatial bases
                start_time = time.time()
                cov_full = self.poisson_sandwich_kron(
                    Z_np, self.X_spatial_array, Y_np, mu_group, meat="cluster"
                )  # (R*P, R*P)
                logger.info("Sandwich cov for group %s computed in %.1fs", group, time.time() - start_time)
                # Extract diagonal blocks: cov_beta[s] = cov_full[s*P:(s+1)*P, s*P:(s+1)*P]
                all_cov_beta[group] = [
                    cov_full[s * P:(s + 1) * P, s * P:(s + 1) * P]
                    for s in range(n_cov)
                ]
                del Z_np, Y_np, mu_group, cov_full
        print("Variance of beta computed")

        # --- Variance of the test statistic ---
        if self.n_group == 1:
            # Homogeneity: variance from non-intercept covariates only.
            # Var(X @ beta_s) = X @ Cov(beta_s) @ X^T  (no bar_Z factor)
            group = self.group_names[0]
            n_cov = self.n_covariates[group]
            var_total = np.zeros(self.n_voxel)
            for s in range(n_cov - 1):  # exclude intercept (last column)
                start_time = time.time()
                var_s = np.einsum(
                    'ij,jk,ik->i', self.X_spatial_array,
                    all_cov_beta[group][s], self.X_spatial_array
                )
                var_total += var_s
                print(f"Variance for covariate {s} in {group}: {time.time() - start_time:.2f}s")
            del all_cov_beta
            contrast_var_bar_eta = var_total.reshape(1, -1)  # (1, n_voxel)
        else:
            # Group comparison: variance from all covariates per group, then contrast
            all_var_bar_eta = {}
            for group in self.group_names:
                var_bar_eta = []
                for s in range(self.n_covariates[group]):
                    start_time = time.time()
                    var_s = all_bar_Z[group][s] * np.einsum(
                        'ij,jk,ik->i', self.X_spatial_array,
                        all_cov_beta[group][s], self.X_spatial_array
                    )
                    print(f"Variance for cov {s} in {group}: {time.time() - start_time:.2f}s")
                    var_bar_eta.append(var_s)
                var_bar_eta = np.stack(var_bar_eta, axis=0)  # (n_cov, n_voxel)
                all_var_bar_eta[group] = var_bar_eta
            del all_cov_beta
            a = np.concatenate([
                np.sum(all_var_bar_eta[group], axis=0).reshape(1, -1)
                for group in self.group_names
            ], axis=0)  # (n_group, n_voxel)
            logger.info("Aggregated variance shape: %s", a.shape)
            contrast_var_bar_eta = self.contrast_vector ** 2 @ a  # (n_contrast, n_voxel)

        contrast_std_bar_eta = np.sqrt(contrast_var_bar_eta)
        # Conduct Wald test (two-sided Z test)
        z_stats = contrast_eta / contrast_std_bar_eta  # (n_contrast, n_voxel)
        p_vals = 2.0 * scipy.stats.norm.sf(np.abs(z_stats))  # two-sided p-values
        logger.info(
            "p-values: min=%.4g, max=%.4g, significant=%d, shape=%s",
            np.min(p_vals), np.max(p_vals),
            np.count_nonzero(p_vals < 0.05), p_vals.shape,
        )
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
                    nll = lambda beta, g=group: SpatialBrainLesionModel._neg_log_likelihood(
                        self.marginal_dist, self.link_func, self.regression_terms,
                        self.X_spatial, self.Y[g], self.Z[g], beta, self.device)
                    H = torch.autograd.functional.hessian(nll, self.beta, create_graph=False)
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

    def poisson_sandwich_kron(self, Z, B, y, mu, *, meat="cluster", ridge=0.0):
        """Memory-efficient sandwich covariance for Poisson log-link GLM.

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

        r = y - mu  # (M, N)

        # Bread: A (p x p)
        w_bread = np.einsum('ik,il,ij->klj', Z, Z, mu)  # (R, R, N)
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
        # load X_spatial, G, P, Y
        self.G = data["G"].item()
        # group names
        self.group_names = list(self.G.keys())
        self.n_subject_per_group = [len(self.G[group]) for group in self.group_names]
        self.n_group = len(self.group_names)
        # load P, Y, Z
        self.B = np.concatenate([data["X_spatial"], np.ones((data["X_spatial"].shape[0], 1))], axis=1)
        self.Z = np.concatenate([data["Z"], np.ones((data["Z"].shape[0], 1))], axis=1)
        self._M, self._R = self.Z.shape
        self._N, self._P = self.B.shape
        # Load parameters and re-scale
        self.beta = params["beta"]
        # self.MU = compute_mu(self.rescaled_Z, self.rescaled_B, self.beta, mode="dask", block_size=1000) # shape: (n_subject*n_voxel, 1)
        self.MU = compute_mu(self.Z, self.B, self.beta, mode="dask", block_size=1000) # shape: (n_subject*n_voxel, 1)
        self.Y = data["Y"] # shape: (n_subject, n_voxel)
        self.Y_reshape = self.Y.reshape(-1, 1) # shape: (n_subject*n_voxel, 1)
    
    def create_contrast(self, contrast_vector=None, contrast_name=None):
        self.contrast_vector = contrast_vector
        self.contrast_name = contrast_name
        # Preprocess the contrast vector
        self.contrast_vector = (
            np.eye(self.n_group)
            if contrast_vector is None
            else np.array(contrast_vector).reshape(1, -1)
        )
        # raise error if dimension of contrast vector doesn't match with number of groups
        if self.contrast_vector.shape[1] != self._R:
            raise ValueError(
                f"""The shape of contrast vector: {str(self.contrast_vector)}
                doesn't match with number of groups."""
            )
        # standardization (row sum 1)
        self.contrast_vector = self.contrast_vector / np.sum(np.abs(self.contrast_vector), axis=1).reshape((-1, 1))
        
    def run_inference(self, method="FI", fig_filename=None):
        # Generalised linear hypothesis testing
        p_vals = self._glh_con_group(method)
        # Plot the estimated P, standard error of P, and p-values
        print(fig_filename)
        if fig_filename is not None:
            self.plot_1d(p_vals, fig_filename, 0.05)

    def _glh_con_group(self, method, use_dask=True, batch_size=20):
        bar_Z = np.mean(self.Z, axis=0) # shape: (n_covariates,)
        # scale the contrast vector by the group size
        bar_Z_reshape = bar_Z.reshape(1, -1)
        group_n_subjects = bar_Z_reshape[:, -self.n_group-1:-1]
        group_ratio = group_n_subjects / np.max(group_n_subjects)
        self.contrast_vector[:, -self.n_group-1:-1] /= group_ratio
        # bar_eta_covariates
        beta_reshape = self.beta.reshape(self._P, self._R, order="F")
        # bar_eta_covariates = (self.B @ beta_reshape).T * bar_Z[:,np.newaxis] # shape: (n_covariates, n_voxel)
        bar_eta_covariates = (bar_Z * beta_reshape).T @ self.B.T # shape: (n_covariates, n_voxel)
        contrast_eta_covariates = self.contrast_vector @ bar_eta_covariates # shape: (1, n_voxel)
        print(np.min(contrast_eta_covariates), np.max(contrast_eta_covariates))
        del bar_eta_covariates
        # Estimate the variance of beta, from either FI or sandwich estimator
        start_time = time.time()
        if method == "FI":
            F_beta = efficient_kronT_diag_kron(self.Z, self.B, self.MU, use_dask=use_dask, block_size=1e4) # shape: (n_covariates*n_bases, n_covariates*n_bases)
            cov_beta = [robust_inverse(F_beta[i*self._P:(i+1)*self._P, i*self._P:(i+1)*self._P]+1e-6*np.eye(self._P)) for i in range(self._R)]
            del F_beta
            print(f"Time taken for Fisher Information: {time.time() - start_time}")
        elif method == "sandwich":
            bread_term = self.bread_term(self.Z, self.B, self.MU) # list: len = n_covariates
            meat_term = self.meat_term(self.Z, self.B, self.MU, self.Y_reshape) # list: len = n_covariates
            # # sandwich estimator
            cov_beta = [B @ M @ B for B, M in zip(bread_term, meat_term)]
            print(len(cov_beta))
            print(np.min(np.diag(cov_beta[1])), np.mean(np.diag(cov_beta[1])), np.max(np.diag(cov_beta[1])))
            print(np.min(np.diag(cov_beta[2])), np.mean(np.diag(cov_beta[2])), np.max(np.diag(cov_beta[2])))
            print("====================================")
            del bread_term, meat_term
            print(f"Time taken for sandwich estimator: {time.time() - start_time}")
        print("Variance of beta computed")
        var_bar_eta = list()
        for s in range(self._R):
            # for covariate s, at voxel j
            # bar_eta_sj = bar_Z_s * X_j^T @ beta_s -- dim: (1,)
            # COV(bar_eta_sj) = bar_Z_s * X_j^T @ COV(beta_s) @ X_j -- dim: (1,)
            # COV(bar_eta_s) = bar_Z_s**2 * X @ COV(beta_s) @ X^T -- dim: (n_voxel, n_voxel)
            var_bar_eta_s = bar_Z[s] * np.einsum('ij,jk,ik->i', self.B, cov_beta[s], self.B) # shape: (n_voxel,)
            var_bar_eta.append(var_bar_eta_s)
            del var_bar_eta_s
        var_bar_eta = np.stack(var_bar_eta, axis=0) # shape: (n_covariate, n_voxel)
        print(np.min(var_bar_eta[1]), np.mean(var_bar_eta[1]), np.max(var_bar_eta[1]))
        print(np.min(var_bar_eta[2]), np.mean(var_bar_eta[2]), np.max(var_bar_eta[2]))
        del cov_beta
        gc.collect()
        # Compute the numerator of the Z test
        contrast_var_bar_eta = self.contrast_vector**2 @ var_bar_eta # shape: (1, n_voxel)
        contrast_std_bar_eta = np.sqrt(contrast_var_bar_eta) # shape: (1, n_voxel)
        # Conduct Wald test (Z test)
        z_stats_eta = contrast_eta_covariates / contrast_std_bar_eta
        print(np.min(z_stats_eta), np.max(z_stats_eta))
        z_stats = np.concatenate([z_stats_eta, -z_stats_eta], axis=0) # shape: (2, n_voxel)
        p_vals = scipy.stats.norm.sf(z_stats) # shape: (2, n_voxel)
        print(np.min(p_vals), np.mean(p_vals), np.max(p_vals))
        exit()
        return p_vals
    
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