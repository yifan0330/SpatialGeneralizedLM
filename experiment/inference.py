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

class BrainInference:
    """Unified public interface for brain-lesion inference.

    Parameters
    ----------
    inference_type : {"full", "approximate", "ukb"}
        Selects the inference backend. The public API is intentionally the
        same for all backends: call ``load_params()``, ``create_contrast()``,
        then ``run_inference()``.
    """

    _BACKENDS = {
        "full": "_FullInferenceBackend",
        "approximate": "_ApproximateInferenceBackend",
        "ukb": "_UKBInferenceBackend",
    }

    def __init__(self, model, marginal_dist, link_func, regression_terms,
                 inference_type="full", space_dim=None, random_seed=None,
                 fewer_voxels=False, dtype=torch.float64, device='cpu'):
        if inference_type not in self._BACKENDS:
            valid = ", ".join(sorted(self._BACKENDS))
            raise ValueError(f"Unknown inference_type={inference_type!r}. Expected one of: {valid}.")

        backend_cls = globals()[self._BACKENDS[inference_type]]
        backend_kwargs = dict(
            model=model,
            marginal_dist=marginal_dist,
            link_func=link_func,
            regression_terms=regression_terms,
            dtype=dtype,
            device=device,
        )
        if inference_type == "full":
            if space_dim is None or random_seed is None:
                raise ValueError("space_dim and random_seed are required for full inference.")
            backend_kwargs.update(
                space_dim=space_dim,
                random_seed=random_seed,
                fewer_voxels=fewer_voxels,
            )

        self.inference_type = inference_type
        self._backend = backend_cls(**backend_kwargs)

    def __getattr__(self, name):
        """Delegate backend-specific attributes for backward compatibility."""
        return getattr(self._backend, name)

    def load_params(self, data, params):
        return self._backend.load_params(data=data, params=params)

    def create_contrast(self, contrast_vector=None, contrast_name=None, polynomial_order=None):
        return self._backend.create_contrast(
            contrast_vector=contrast_vector,
            contrast_name=contrast_name,
            polynomial_order=polynomial_order,
        )

    def run_inference(self, *args, **kwargs):
        return self._backend.run_inference(*args, **kwargs)


class _BaseInferenceBackend(object):
    """Shared utilities for concrete inference backends."""

    def __init__(self, model, marginal_dist, link_func, regression_terms,
                 dtype=torch.float64, device='cpu'):
        self.model = model
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.dtype = dtype
        self.device = device

    @staticmethod
    def _extract_param(params, key):
        """Return an NPZ/dict parameter value, unwrapping object scalars."""
        value = params[key]
        return value.item() if getattr(value, "ndim", 1) == 0 else value

    @staticmethod
    def _with_intercept(array):
        """Append an intercept column to a 2D design matrix."""
        return np.concatenate([array, np.ones((array.shape[0], 1))], axis=1)

    def _scaled_with_intercept(self, array, scale=50.0):
        """Scale a design matrix by ``scale / n_rows`` and append intercept."""
        return self._with_intercept(array * scale / array.shape[0])

    @staticmethod
    def _is_multigroup_data(data):
        """Return True for the object-array group format used by simulations."""
        first_value = next(iter(data.values()))
        return (
            hasattr(first_value, "ndim")
            and first_value.ndim == 0
            and hasattr(first_value, "item")
            and isinstance(first_value.item(), dict)
            and {"Y", "Z", "X_spatial"}.issubset(first_value.item())
        )

    def _normalise_contrast(self, contrast_vector, expected_width):
        """Validate and L1-normalise contrast rows."""
        contrast_vector = np.asarray(contrast_vector, dtype=float)
        if contrast_vector.ndim == 1:
            contrast_vector = contrast_vector.reshape(1, -1)
        if contrast_vector.shape[1] != expected_width:
            raise ValueError(
                f"Contrast vector shape {contrast_vector.shape} doesn't match "
                f"expected width ({expected_width})."
            )
        row_norm = np.sum(np.abs(contrast_vector), axis=1, keepdims=True)
        if np.any(row_norm == 0):
            raise ValueError("Contrast rows must contain at least one non-zero entry.")
        return contrast_vector / row_norm

    @staticmethod
    def _default_group_contrast(n_group):
        """Default contrast for group-level inference."""
        if n_group == 1:
            return np.eye(1)
        contrast = np.zeros((n_group - 1, n_group))
        for k in range(n_group - 1):
            contrast[k, k] = 1
            contrast[k, k + 1] = -1
        return contrast

    @staticmethod
    def _default_covariate_contrast(n_covariates, index=0):
        """Default contrast selecting one covariate column."""
        contrast = np.zeros((1, n_covariates))
        contrast[0, index] = 1
        return contrast

    @staticmethod
    def _ukb_age_contrast(n_covariates, polynomial_order=1):
        """Age contrast rows for the UKB covariate layout."""
        if polynomial_order == 1:
            contrast = np.zeros((1, n_covariates))
            contrast[0, 1] = 1
            return contrast

        contrast = np.zeros((3, n_covariates))
        contrast[0, 2] = 1
        contrast[1, 3] = 1
        contrast[2, 4] = 1
        return contrast

    def _set_contrast(self, contrast_vector, expected_width, contrast_name=None):
        """Store a validated, normalised contrast matrix."""
        self.contrast_name = contrast_name
        self.contrast_vector = self._normalise_contrast(contrast_vector, expected_width)
        self._S = self.contrast_vector.shape[0]
        return self.contrast_vector

    def _set_matrix_dimensions(self, spatial_attr="B", covariate_attr="Z"):
        """Populate common matrix dimensions from loaded design matrices."""
        Z = getattr(self, covariate_attr)
        B = getattr(self, spatial_attr)
        self._M, self._R = Z.shape
        self._N, self._P = B.shape

    def _load_flat_design(self, data, *, scale=50.0):
        """Load flat ``{X_spatial, Z, Y}`` arrays into numpy attributes."""
        self.group_names = ["Group_1"]
        self.n_group = 1
        self.n_subject = {"Group_1": data["Y"].shape[0]}
        self.B = self._scaled_with_intercept(data["X_spatial"], scale=scale)
        self.Z = self._scaled_with_intercept(data["Z"], scale=scale)
        self.Y = data["Y"]
        self._set_matrix_dimensions()

    def _load_stacked_multigroup_design(self, data, *, scale=50.0):
        """Load and stack multi-group simulation arrays for approximate inference."""
        self.group_names = list(data.keys())
        self.n_group = len(self.group_names)
        self.n_subject = {}

        first_group = self.group_names[0]
        X_spatial = data[first_group].item()["X_spatial"]
        self.B = self._scaled_with_intercept(X_spatial, scale=scale)

        Y_all, Z_all = [], []
        for group_name in self.group_names:
            group_data = data[group_name].item()
            Y_g = group_data["Y"]
            Z_g = group_data["Z"]
            self.n_subject[group_name] = Y_g.shape[0]
            Y_all.append(Y_g)
            Z_all.append(Z_g)

        Z_cat = np.concatenate(Z_all, axis=0)
        self.Z = self._scaled_with_intercept(Z_cat, scale=scale)
        self.Y = np.concatenate(Y_all, axis=0)
        self._set_matrix_dimensions()

    def _load_shared_beta(self, params, *, reject_dict=False):
        """Load a shared beta parameter from params."""
        beta = self._extract_param(params, "beta")
        if reject_dict and isinstance(beta, dict):
            raise NotImplementedError(
                "This inference backend does not support per-group beta dict. "
                "Use full-model inference or provide a shared beta array."
            )
        self.beta = beta
        return beta

    def _load_probability_mean(self, params, group_names):
        """Load fitted probabilities and store their voxel-wise mean."""
        P_value = self._extract_param(params, "P")
        if isinstance(P_value, dict):
            self.P_mean = np.stack(
                [np.mean(P_value[group], axis=0) for group in group_names], axis=0,
            )
        else:
            self.P_mean = np.mean(P_value, axis=0, keepdims=True)
        self.eta = np.log(self.P_mean)
        return self.P_mean

    def _load_group_or_shared_beta(self, params, group_names, *, as_tensor=False):
        """Load beta as per-group or shared parameters and compatibility attrs."""
        beta_value = self._extract_param(params, "beta")
        if isinstance(beta_value, dict):
            if as_tensor:
                self.beta_dict = {g: torch.tensor(beta_value[g], **self._kwargs) for g in group_names}
            else:
                self.beta_dict = {g: beta_value[g] for g in group_names}
            self.beta_array_dict = {g: beta_value[g] for g in group_names}
            first = group_names[0]
            self.beta = self.beta_dict[first]
            self.beta_array = self.beta_array_dict[first]
        else:
            self.beta = torch.tensor(beta_value, **self._kwargs) if as_tensor else beta_value
            self.beta_array = beta_value
            self.beta_dict = {g: self.beta for g in group_names}
            self.beta_array_dict = {g: self.beta_array for g in group_names}
        return self.beta

    def _compute_mu_matrix(self, Z, B, beta, *, block_size=1000):
        """Compute fitted means and reshape them to ``(n_subject, n_voxel)``."""
        mu = compute_mu(Z, B, beta, mode="dask", block_size=block_size)
        return mu.reshape(Z.shape[0], B.shape[0])

    def _save_npz(self, filename, **arrays):
        """Save arrays to an NPZ file, creating the parent directory if needed."""
        if filename is None:
            return
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.savez(filename, **arrays)

    def _load_or_compute_array(self, filename, key, compute_fn,
                               *, allow_pickle=False, load_message=None,
                               compute_message=None):
        """Shared cache helper for one-array NPZ files."""
        if filename is not None and os.path.exists(filename):
            if load_message:
                print(load_message)
            return np.load(filename, allow_pickle=allow_pickle)[key]
        if compute_message:
            print(compute_message)
        value = compute_fn()
        if filename is not None:
            self._save_npz(filename, **{key: value})
        return value

    def _load_or_compute_npz_dict(self, filename, compute_fn, *, allow_pickle=False):
        """Shared cache helper for multi-array NPZ files."""
        if filename is not None and os.path.exists(filename):
            loaded = np.load(filename, allow_pickle=allow_pickle)
            return {key: loaded[key] for key in loaded.files}
        result = compute_fn()
        if filename is not None:
            self._save_npz(filename, **result)
        return result

    def _load_or_compute_inference(self, compute_fn, *, inference_filename=None,
                                   p_vals_filename=None, z_vals_filename=None,
                                   log_prefix="INFERENCE"):
        """Shared cache handling for inference outputs."""
        if inference_filename is not None and os.path.exists(inference_filename):
            print(f"[{log_prefix}] LOADING CACHED inference from {inference_filename}")
            loaded = np.load(inference_filename)
            return loaded["p_vals"], loaded["z_stats"]

        if (
            p_vals_filename is not None
            and z_vals_filename is not None
            and os.path.exists(p_vals_filename)
            and os.path.exists(z_vals_filename)
        ):
            print(f"[{log_prefix}] loaded p-values and z-stats from file.")
            return np.load(p_vals_filename)["p_vals"], np.load(z_vals_filename)["z_stats"]

        if inference_filename is not None:
            print(f"[{log_prefix}] Computing fresh inference → {inference_filename}")
        p_vals, z_stats = compute_fn()
        if inference_filename is not None:
            self._save_npz(inference_filename, p_vals=p_vals, z_stats=z_stats)
        if p_vals_filename is not None:
            self._save_npz(p_vals_filename, p_vals=p_vals)
        if z_vals_filename is not None:
            self._save_npz(z_vals_filename, z_stats=z_stats)
        return p_vals, z_stats

    def _log_inference_summary(self, p_vals, z_stats, *, alpha=0.05, two_sided=True):
        """Print a compact summary of inference results."""
        print(f"[INFERENCE] z_stats: min={z_stats.min():.4f}, max={z_stats.max():.4f}, "
              f"mean={z_stats.mean():.4f}, std={np.std(z_stats):.4f}")
        if two_sided:
            significant = np.count_nonzero(2.0 * scipy.stats.norm.sf(np.abs(z_stats)) < alpha)
            print(f"[INFERENCE] significant (two-sided, alpha={alpha}): {significant}/{z_stats.size}")
        else:
            print(f"[INFERENCE] significant (alpha={alpha}): {np.count_nonzero(p_vals < alpha)}/{p_vals.size}")
        logger.info("p_vals shape: %s", p_vals.shape)

    def _plot_brain_z_maps(self, z_stats, fig_filename, lesion_mask, *, alpha=0.05,
                           suffix_multiple=True, output_uncorrected=False):
        """Shared z-statistic brain-map plotting."""
        if fig_filename is None or lesion_mask is None:
            return
        parent = os.path.dirname(fig_filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        z_threshold = scipy.stats.norm.ppf(1 - alpha)
        print("threshold", z_threshold)
        if z_stats.ndim == 2 and z_stats.shape[0] > 1 and suffix_multiple:
            for i in range(z_stats.shape[0]):
                out_fname = fig_filename.replace(".png", f"_contrast_{i}.png")
                plot_brain(p=z_stats[i], brain_mask=lesion_mask, threshold=z_threshold, output_filename=out_fname)
        else:
            output_filename = fig_filename.replace(".png", "_uncorrected.png") if output_uncorrected else fig_filename
            plot_brain(p=z_stats.ravel(), brain_mask=lesion_mask, threshold=z_threshold, vmax=None, output_filename=output_filename)

    def _save_brain_outputs(self, p_vals, z_stats, lesion_mask, fig_dir, method):
        """Save p-value and z-statistic NIfTI maps."""
        if lesion_mask is None or fig_dir is None:
            return
        os.makedirs(fig_dir, exist_ok=True)
        save_nifti(p_vals.flatten(), lesion_mask, os.path.join(fig_dir, f"p_vals_{self.model}_{method}.nii.gz"))
        save_nifti(z_stats.flatten(), lesion_mask, os.path.join(fig_dir, f"z_stats_{self.model}_{method}.nii.gz"))

    def poisson_sandwich_kron(self,
                              Z,
                              B,
                              y,
                              mu,
                              *,
                              meat="cluster",
                              ridge=0.0,
                              bread_weights=None,
                              score_residuals=None,
                              mu_for_meat=None,
                              correction=None,
                              return_diagnostics=False):
        """Memory-efficient sandwich covariance for Kronecker-design GLMs."""
        Z = np.asarray(Z, dtype=float)
        B = np.asarray(B, dtype=float)
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)

        M, R = Z.shape
        N, P = B.shape
        p = R * P

        if y.shape != (M, N) or mu.shape != (M, N):
            raise ValueError("y and mu must have shape (M, N) matching Z and B.")
        if not np.isfinite(Z).all() or not np.isfinite(B).all() or not np.isfinite(y).all():
            raise ValueError("Inputs Z/B/y contain NaN or Inf.")

        mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=0.0)
        mu = np.clip(mu, 1e-12, 1e12)

        if bread_weights is None:
            bread_weights = mu
        else:
            bread_weights = np.asarray(bread_weights, dtype=float)
            bread_weights = np.nan_to_num(bread_weights, nan=0.0, posinf=1e12, neginf=0.0)
            bread_weights = np.clip(bread_weights, 1e-12, 1e12)

        if score_residuals is not None:
            r = np.asarray(score_residuals, dtype=float)
        elif mu_for_meat is not None:
            mu_r = np.asarray(mu_for_meat, dtype=float)
            mu_r = np.nan_to_num(mu_r, nan=0.0, posinf=1e12, neginf=0.0)
            mu_r = np.clip(mu_r, 1e-12, 1e12)
            r = y - mu_r
        else:
            r = y - mu
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

        w_bread = np.einsum('ik,il,ij->klj', Z, Z, bread_weights)
        A = np.zeros((p, p))
        for k in range(R):
            for k2 in range(k, R):
                block = B.T @ (B * w_bread[k, k2, :, None])
                A[k * P:(k + 1) * P, k2 * P:(k2 + 1) * P] = block
                if k != k2:
                    A[k2 * P:(k2 + 1) * P, k * P:(k + 1) * P] = block

        if ridge > 0:
            A += ridge * np.eye(p)

        A = np.nan_to_num(A, nan=0.0, posinf=1e12, neginf=-1e12)
        A = 0.5 * (A + A.T)

        hc_factor = self._apply_sandwich_correction(
            correction, A, Z, B, bread_weights, r, ridge, M, R, P, p
        )
        if isinstance(hc_factor, tuple):
            hc_factor, r = hc_factor

        C, Bmeat, meat_kind = self._sandwich_meat(Z, B, r, meat, M, R, P, p)

        try:
            ridge_eps = max(ridge, 1e-8)
            L, low = scipy.linalg.cho_factor(A + ridge_eps * np.eye(p))
            if meat_kind == "cluster":
                Y = scipy.linalg.cho_solve((L, low), C)
                cov = Y @ Y.T
            else:
                D = scipy.linalg.cho_solve((L, low), Bmeat)
                cov = scipy.linalg.cho_solve((L, low), D.T).T
        except (np.linalg.LinAlgError, ValueError):
            logger.warning("Cholesky failed — falling back to pseudo-inverse")
            Ainv = np.linalg.pinv(A + 1e-6 * np.eye(p))
            if meat_kind == "cluster":
                Y = Ainv @ C
                cov = Y @ Y.T
            else:
                cov = Ainv @ np.nan_to_num(Bmeat, nan=0.0, posinf=1e12, neginf=-1e12) @ Ainv

        cov *= hc_factor
        cov = 0.5 * (cov + cov.T)

        if return_diagnostics:
            return cov, {
                "method": "kron_sandwich",
                "meat": meat_kind,
                "ridge": ridge,
                "correction": correction,
                "M": M,
                "N": N,
                "R": R,
                "P": P,
                "p": p,
            }
        return cov

    def _apply_sandwich_correction(self, correction, A, Z, B, bread_weights, r, ridge, M, R, P, p):
        if correction is None:
            return 1.0

        correction_kind = correction.lower()
        if correction_kind == "hc0":
            return 1.0
        if correction_kind == "hc1":
            if M <= R:
                raise ValueError(
                    f"HC1 correction requires M > R, got M={M}, R={R}. "
                    "Use correction='hc0' for this setting."
                )
            hc_factor = M / float(M - R)
            logger.info("Applying HC1 sandwich meat correction: M/(M-R)=%.6g", hc_factor)
            return hc_factor
        if correction_kind != "hc3":
            raise ValueError("correction must be None, 'hc0', 'hc1', or 'hc3'.")

        ridge_eps = max(ridge, 1e-8)
        try:
            L_h, low_h = scipy.linalg.cho_factor(A + ridge_eps * np.eye(p))
            Ainv_for_h = scipy.linalg.cho_solve((L_h, low_h), np.eye(p))
        except (np.linalg.LinAlgError, ValueError):
            logger.warning("Cholesky failed while computing HC3 leverage — using pseudo-inverse")
            Ainv_for_h = np.linalg.pinv(A + 1e-6 * np.eye(p))

        Ainv_blocks = Ainv_for_h.reshape(R, P, R, P).transpose(0, 2, 1, 3)
        leverage_basis = np.einsum('jp,klpq,jq->klj', B, Ainv_blocks, B)
        hii = bread_weights * np.einsum('ik,il,klj->ij', Z, Z, leverage_basis)
        hii = np.nan_to_num(hii, nan=0.0, posinf=1.0, neginf=0.0)
        hii = np.clip(hii, 0.0, 0.999)
        r = r / np.maximum(1.0 - hii, 1e-6)
        logger.info(
            "Applying HC3 sandwich correction: leverage min/mean/max = %.4g / %.4g / %.4g",
            float(np.min(hii)), float(np.mean(hii)), float(np.max(hii)),
        )
        return 1.0, r

    def _sandwich_meat(self, Z, B, r, meat, M, R, P, p):
        meat_kind = meat.lower()
        if meat_kind == "cluster":
            Bt_r = B.T @ r.T
            C = np.zeros((p, M))
            for k in range(R):
                C[k * P:(k + 1) * P, :] = Bt_r * Z[:, k][None, :]
            return C, None, meat_kind
        if meat_kind == "iid":
            w_meat = np.einsum('ik,il,ij->klj', Z, Z, r ** 2)
            Bmeat = np.zeros((p, p))
            for k in range(R):
                for k2 in range(k, R):
                    block = B.T @ (B * w_meat[k, k2, :, None])
                    Bmeat[k * P:(k + 1) * P, k2 * P:(k2 + 1) * P] = block
                    if k != k2:
                        Bmeat[k2 * P:(k2 + 1) * P, k * P:(k + 1) * P] = block
            return None, Bmeat, meat_kind
        raise ValueError("meat must be 'iid' or 'cluster'.")

    def compute_fisher_information(self, Z, B, mu, *, use_dask=True,
                                   block_size=1e4, cache_filename=None,
                                   cache_key="XTWX"):
        """Compute or load ``X.T @ W @ X`` for the Kronecker design."""
        if cache_filename is not None and os.path.exists(cache_filename):
            return np.load(cache_filename)[cache_key]

        fisher_info = efficient_kronT_diag_kron(
            Z, B, mu, use_dask=use_dask, block_size=block_size,
        )
        if cache_filename is not None:
            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
            np.savez(cache_filename, **{cache_key: fisher_info})
        return fisher_info

    def fisher_covariance(self, fisher_info, *, ridge=1e-6, inverse="pinv"):
        """Invert a Fisher-information matrix with a small ridge term."""
        fisher_info = np.asarray(fisher_info, dtype=float)
        regularized = fisher_info + ridge * np.eye(fisher_info.shape[0])
        if inverse == "inv":
            return np.linalg.inv(regularized)
        if inverse == "pinv":
            return np.linalg.pinv(regularized)
        raise ValueError("inverse must be 'inv' or 'pinv'.")

    def blockwise_fisher_covariance(self, fisher_info, n_blocks, block_size,
                                    *, ridge=1e-6):
        """Return block-diagonal covariance blocks from a Fisher matrix."""
        return [
            robust_inverse(
                fisher_info[i * block_size:(i + 1) * block_size,
                            i * block_size:(i + 1) * block_size]
                + ridge * np.eye(block_size)
            )
            for i in range(n_blocks)
        ]

    def autograd_hessian_covariance(self, hessian, n_covariates, n_bases,
                                    *, ridge=1e-6):
        """Convert an autograd Hessian ``(P, R, P, R)`` to covariance."""
        fisher_full = hessian.transpose(1, 0, 3, 2).reshape(
            n_covariates * n_bases,
            n_covariates * n_bases,
        )
        return self.fisher_covariance(fisher_full, ridge=ridge, inverse="inv")

    def sandwich_covariance(self, Z, B, Y, mu, *, meat="cluster", ridge=0.0,
                            mu_for_meat=None, bread_weights=None,
                            score_residuals=None, correction=None,
                            return_diagnostics=False):
        """Compute sandwich covariance through the shared Kronecker routine."""
        return self.poisson_sandwich_kron(
            Z, B, Y, mu,
            meat=meat,
            ridge=ridge,
            bread_weights=bread_weights,
            score_residuals=score_residuals,
            mu_for_meat=mu_for_meat,
            correction=correction,
            return_diagnostics=return_diagnostics,
        )

    def contrast_design(self, contrast_vector, B):
        """Build flattened ``C ⊗ B`` rows for all contrasts and voxels."""
        contrast_vector = np.asarray(contrast_vector)
        CB = np.einsum('ij,kl->ikjl', contrast_vector, B)
        return CB.reshape(contrast_vector.shape[0], B.shape[0], -1)

    def full_covariance_contrast_variance(self, contrast_design, covariance,
                                          *, keepdims=False):
        """Compute ``diag((C⊗B) Cov(beta) (C⊗B)^T)`` per voxel."""
        tmp = np.einsum('snk,kl->snl', contrast_design, covariance)
        return np.sum(tmp * contrast_design, axis=-1, keepdims=keepdims)

    def blockwise_contrast_variance(self, B, contrast_vector, cov_blocks):
        """Contrast variance when covariance is approximated by covariate blocks."""
        var_eta = []
        for cov_block in cov_blocks:
            var_eta.append(np.einsum('ij,jk,ik->i', B, cov_block, B))
        var_eta = np.stack(var_eta, axis=0)
        return np.asarray(contrast_vector) ** 2 @ var_eta

    def mass_univariate_fisher_covariance(self, Z, beta):
        """Per-voxel Fisher covariance for mass-univariate GLMs."""
        beta = np.asarray(beta)
        if beta.ndim == 2 and beta.shape[1] == Z.shape[1]:
            linear = Z @ beta.T
        elif beta.shape[0] == Z.shape[1]:
            linear = Z @ beta
        else:
            raise ValueError(
                f"Cannot align beta shape {beta.shape} with Z shape {Z.shape}."
            )

        if self.link_func == "log":
            mu = np.exp(linear)
            fisher_info = np.einsum('im,ij,ik->jmk', Z, mu, Z)
        elif self.link_func == "logit":
            mu = 1.0 / (1.0 + np.exp(-linear))
            fisher_info = np.einsum('im,ij,ik->jmk', Z, mu * (1.0 - mu), Z)
        else:
            raise ValueError(f"Link function {self.link_func} not supported.")

        return np.linalg.pinv(fisher_info), mu

    def plot_1d(self, p_vals, filename, significance_level=0.05):
        p_vals = np.atleast_2d(p_vals)
        n_panels, n_voxel = p_vals.shape
        fig, axes = plt.subplots(1, n_panels, figsize=(11.5 * n_panels, 11), squeeze=False)
        axes = axes.ravel()

        th_p = np.arange(1 / float(n_voxel), 1 + 1 / float(n_voxel), 1 / float(n_voxel))
        th_p_log = -np.log10(th_p)
        k_array = np.arange(start=1, stop=n_voxel + 1, step=1)
        ci_lower = scipy.stats.beta.ppf(significance_level / 2, k_array, n_voxel - k_array + 1)
        ci_upper = scipy.stats.beta.ppf(1 - significance_level / 2, k_array, n_voxel - k_array + 1)

        for i, ax in enumerate(axes):
            sorted_p_vals = np.sort(p_vals[i, :])
            significance_percentage = np.sum(sorted_p_vals < significance_level) / n_voxel
            ax.fill_between(
                th_p_log,
                -np.log10(ci_lower),
                -np.log10(ci_upper),
                color='grey',
                alpha=0.5,
                label=f'{int((1 - significance_level) * 100)}% Beta CI',
            )
            ax.plot(th_p_log, np.repeat(-np.log10(significance_level), n_voxel), color='y', linestyle='--', label=f'threshold at -log10({significance_level})')
            ax.plot(th_p_log, -np.log10(th_p), color='orange', linestyle='--', label='y=x')
            ax.plot(th_p_log, -np.log10(significance_level * th_p), color='red', linestyle='-', label='FDR(BH) control')
            ax.scatter(th_p_log, -np.log10(sorted_p_vals), c='#1f77b4', s=4)
            ax.set_xlim([0, np.max(-np.log10(k_array / n_voxel))])
            ax.set_ylim([0, np.max(-np.log10(k_array / n_voxel))])
            ax.set_xlabel("Expected -log10(P)", fontsize=20)
            ax.set_ylabel("Observed -log10(P)", fontsize=20)
            ax.set_title(f"Contrast {i}: {significance_percentage * 100:.2f}% voxels rejected", fontsize=24)
            ax.legend()

        fig.savefig(filename)
        plt.close(fig)

    def histogram_z_stats(self, z_stats, filename):
        plt.figure(figsize=(10, 6))
        plt.hist(np.asarray(z_stats).ravel(), bins=100, color='blue', alpha=0.7, edgecolor='black')
        plt.title('Histogram of Z-statistics', fontsize=16)
        plt.xlabel('Z-statistic', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(filename)
        plt.close()

class _FullInferenceBackend(_BaseInferenceBackend):
    def __init__(self, model,space_dim, marginal_dist, link_func, regression_terms, random_seed, fewer_voxels=False,
                dtype=torch.float64, device='cpu'):
        super().__init__(model, marginal_dist, link_func, regression_terms, dtype=dtype, device=device)
        self.space_dim=space_dim
        self.random_seed = random_seed
        self.fewer_voxels = fewer_voxels
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
        self.X_spatial_array = self._with_intercept(X_spatial)
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
            Z_with_intercept = self._with_intercept(Z_g)
            self.Y[group_name] = torch.tensor(Y_g, **self._kwargs)
            self.Z[group_name] = torch.tensor(Z_with_intercept, **self._kwargs)
            self.n_subject[group_name] = Z_with_intercept.shape[0]
            self.n_covariates[group_name] = Z_with_intercept.shape[1]

        self._load_probability_mean(params, self.group_names)
        self._load_group_or_shared_beta(params, self.group_names, as_tensor=True)
        self.n_voxel, self.n_bases = self.X_spatial.shape

    def create_contrast(self, contrast_vector=None, contrast_name=None, polynomial_order=None):
        """Build and normalise the contrast vector over groups."""
        if contrast_vector is None:
            contrast_vector = self._default_group_contrast(self.n_group)
        else:
            contrast_vector = np.array(contrast_vector).reshape(1, -1)
        self._set_contrast(contrast_vector, self.n_group, contrast_name)

    def run_inference(self, method="FI", inference_filename=None, fig_filename=None, lesion_mask=None, alpha=0.05):
        p_vals, z_stats = self._load_or_compute_inference(
            lambda: self._glh_con_group(method),
            inference_filename=inference_filename,
        )
        self._log_inference_summary(p_vals, z_stats, alpha=alpha, two_sided=True)
        logger.info("Plotting inference results to %s", fig_filename)
        self._plot_brain_z_maps(z_stats, fig_filename, lesion_mask, alpha=alpha)

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

        Cov_beta, _ = self.mass_univariate_fisher_covariance(Z_np, self.beta_array)

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
                all_cov_beta[group] = self.autograd_hessian_covariance(
                    F_beta, n_cov, P_dim, ridge=1e-6,
                )
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
                cov_full = self.sandwich_covariance(
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

        def compute_hessians():
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
            return all_H

        return self._load_or_compute_npz_dict(
            Fisher_info_filename, compute_hessians, allow_pickle=True,
        )

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
    
class _ApproximateInferenceBackend(_BaseInferenceBackend):
    def __init__(self, model, marginal_dist, link_func, regression_terms, 
                dtype=torch.float64, device='cpu'):
        super().__init__(model, marginal_dist, link_func, regression_terms, dtype=dtype, device=device)
    
    def load_params(self, data, params):
        # Support both legacy flat format and multi-group object-array format.
        if self._is_multigroup_data(data):
            self._load_stacked_multigroup_design(data)
        else:
            self._load_flat_design(data)
        # Load parameters and re-scale
        self._load_shared_beta(params, reject_dict=True)
        # self.MU = compute_mu(self.rescaled_Z, self.rescaled_B, self.beta, mode="dask", block_size=1000) # shape: (n_subject*n_voxel, 1)
        self.MU = compute_mu(self.Z, self.B, self.beta, mode="dask", block_size=1000) # shape: (n_subject*n_voxel, 1)
        self.Y_reshape = self.Y.reshape(-1, 1) # shape: (n_subject*n_voxel, 1)
    
    def create_contrast(self, contrast_vector=None, contrast_name=None, polynomial_order=None):
        # Preprocess the contrast vector
        if contrast_vector is None:
            # Default: test the first non-intercept covariate (e.g. age).
            # Z columns are [cov_0 (age), ..., cov_{R-2}, intercept].
            # Contrast [1, 0, ..., 0] tests only the first covariate.
            contrast_vector = self._default_covariate_contrast(self._R, index=0)
        else:
            contrast_vector = np.array(contrast_vector).reshape(1, -1)
        self._set_contrast(contrast_vector, self._R, contrast_name)
        
    def run_inference(self, method="FI", inference_filename=None, fig_filename=None):
        # Generalised linear hypothesis testing
        p_vals, z_stats = self._load_or_compute_inference(
            lambda: self._glh_con_group(method),
            inference_filename=inference_filename,
        )
        # Plot the estimated P, standard error of P, and p-values
        if fig_filename is not None:
            self.plot_1d(p_vals, fig_filename, 0.05)

    def _glh_con_group(
        self,
        method,
        use_dask=True,
        batch_size=20,
        sandwich_meat="null_cluster",
        sandwich_correction="hc3",
    ):
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
            fisher_info = self.compute_fisher_information(
                self.Z, self.B, self.MU, use_dask=use_dask, block_size=1e4,
            )
            cov_beta = self.blockwise_fisher_covariance(
                fisher_info, self._R, self._P, ridge=1e-6,
            )
            del fisher_info
            logger.info("Fisher Information computed in %.1fs", time.time() - start_time)
        elif method == "sandwich":
            MU_matrix = self.MU.reshape(self._M, self._N)
            # null_cluster / null_iid: score-test sandwich.
            # Zero out the contrast covariate beta block, recompute mu_null,
            # use r_null = Y - mu_null in the meat.  E[r_null] = 0 under H0
            # by construction, eliminating inflation from approximate convergence.
            if sandwich_meat in ("null_cluster", "null_iid"):
                beta_null = self.beta.copy().reshape(self._P, self._R, order="F")  # (P, R)
                for s in range(self.contrast_vector.shape[0]):
                    for r_idx in range(self._R):
                        if self.contrast_vector[s, r_idx] != 0:
                            beta_null[:, r_idx] = 0.0
                beta_null_flat = beta_null.reshape(-1, 1, order="F")
                MU_null = compute_mu(
                    self.Z, self.B, beta_null_flat, mode="dask", block_size=1000
                ).reshape(self._M, self._N)
                logger.info("Null mu computed for score-test sandwich (zeroed contrast covariate block)")
                base_meat = sandwich_meat[len("null_"):]  # 'cluster' or 'iid'
                cov_beta_full = self.sandwich_covariance(
                    self.Z, self.B, self.Y, MU_matrix,
                    meat=base_meat,
                    ridge=0.0,
                    mu_for_meat=MU_null,
                    correction=sandwich_correction,
                )  # (R*P, R*P)
            else:
                cov_beta_full = self.sandwich_covariance(
                    self.Z,
                    self.B,
                    self.Y,
                    MU_matrix,
                    meat=sandwich_meat,
                    ridge=0.0,
                    correction=sandwich_correction,
                )  # (R*P, R*P)
            logger.info(
                "Sandwich estimator (%s, %s) computed in %.1fs",
                sandwich_meat,
                sandwich_correction,
                time.time() - start_time,
            )

        if method == "FI":
            # Var(c[s] * B_j @ beta_s) = c[s]^2 * B_j^T Cov(beta_s) B_j  (block-diagonal approx)
            contrast_var_bar_eta = self.blockwise_contrast_variance(
                self.B, self.contrast_vector, cov_beta,
            )
            del cov_beta
            gc.collect()
        else:
            # Full sandwich variance: Var(c @ beta_map_j) = (c ⊗ B_j)^T Cov(beta) (c ⊗ B_j)
            # where c is the contrast over covariates (no bar_Z weighting)
            CB_flat = self.contrast_design(self.contrast_vector, self.B)
            contrast_var_bar_eta = self.full_covariance_contrast_variance(
                CB_flat, cov_beta_full,
            )

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

    def bread_term(self, Z, B, P, use_dask=True, block_size=1000):
        fisher_info = self.compute_fisher_information(
            Z, B, P, use_dask=use_dask, block_size=block_size,
        )
        bread_term = self.blockwise_fisher_covariance(
            fisher_info, self._R, self._P, ridge=1e-6,
        )
        del fisher_info
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

class _UKBInferenceBackend(_BaseInferenceBackend):
    def __init__(self, model, marginal_dist, link_func, regression_terms, 
                dtype=torch.float64, device='cpu'):
        super().__init__(model, marginal_dist, link_func, regression_terms, dtype=dtype, device=device)

    def load_params(self, data, params):
        # Load data
        self._load_flat_design(data)
        # Load parameters and re-scale
        self._load_shared_beta(params)
        # MU
        if self.model == "SpatialBrainLesion":
            self.MU = self._compute_mu_matrix(self.Z, self.B, self.beta, block_size=5000)
            P = self.MU * np.exp(-self.MU) # shape: (n_subject, n_voxel)
            P_mean = np.mean(P, axis=0) # shape: (n_voxel,)

    def create_contrast(self, contrast_vector=None, contrast_name=None, polynomial_order=1):
        # Preprocess the contrast vector
        if contrast_name == "age":
            contrast_vector = self._ukb_age_contrast(self._R, polynomial_order)
        else:
            contrast_vector = (
                np.eye(self._R)
                if contrast_vector is None
                else np.array(contrast_vector).reshape(-1, self._R)
            )
        self._set_contrast(contrast_vector, self._R, contrast_name)

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
        # Generalised linear hypothesis testing
        def compute():
            if self.model == "SpatialBrainLesion":
                return self.SpatialGLM_glh_con_group(method, lesion_mask, True, 1e4)
            elif self.model == "MassUnivariateRegression":
                return self.MUM_glh_con_group(lesion_mask)
            raise ValueError(f"Model {self.model} not supported for inference.")

        p_vals, z_stats = self._load_or_compute_inference(
            compute,
            p_vals_filename=self.p_vals_filename,
            z_vals_filename=self.z_vals_filename,
        )
        # Plot the estimated P, standard error of P, and p-values
        self.histogram_z_stats(z_stats, fig_filename.replace(".png", "_z_stats_histogram.png"))
        self._save_brain_outputs(p_vals, z_stats, lesion_mask, self.fig_dir, method)
        z_threshold = scipy.stats.norm.ppf(1-alpha)
        print(z_threshold, "z_threshold for alpha=", alpha)
        print(np.count_nonzero(p_vals < alpha), p_vals.size, "significant voxels at alpha=", alpha)
        print(np.count_nonzero(z_stats > z_threshold), z_stats.size, "voxels with z_stat > z_threshold at alpha=", alpha)
        self._plot_brain_z_maps(
            z_stats, fig_filename, lesion_mask, alpha=alpha,
            suffix_multiple=False, output_uncorrected=True,
        )
        
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
            XTWX = self.compute_fisher_information(
                self.Z,
                self.B,
                self.MU,
                use_dask=use_dask,
                block_size=block_size,
                cache_filename=self.XTWX_filename,
                cache_key="XTWX",
            )

        CB = np.einsum('ij,kl->ikjl', self.contrast_vector, self.B) # shape: (_S, _N, _R, _P)
        CB_flat = CB.reshape(self._S, self._N, -1) # shape: (_S, _N, _R*_P)
        # (C \otimes B) \beta
        CB_beta = CB_flat @ self.beta  # shape: (_S, _N, 1)
        CB_beta = CB_beta.squeeze(-1) # shape: (_S, _N)
        # get the path of self.fig_filename
        plot_brain(p=CB_beta.flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename=os.path.join(self.fig_dir, "numerator_map_SGLM.png"))
        # shape: (_S, _N) 
        if method == "FI":
            cov_beta = self.fisher_covariance(XTWX, ridge=0.0, inverse="pinv")
            contrast_var_eta = self.full_covariance_contrast_variance(
                CB_flat, cov_beta, keepdims=True,
            )
            plot_brain(p=np.sqrt(contrast_var_eta).flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename=os.path.join(self.fig_dir, "denominator_map_SGLM_FI.png"))
            # del bread_term, meat_term, cov_beta
        elif method == "sandwich":
            cov_beta, diag = self.sandwich_covariance(
                self.Z, self.B, self.Y, self.MU,
                meat="iid", ridge=0, return_diagnostics=True,
            )
            print(np.min(np.diag(cov_beta)), np.mean(np.diag(cov_beta)), np.max(np.diag(cov_beta)), "cov_beta diag stats")
            # meat_term = self.meat_term(self.Z, self.B, self.MU, self.Y) 
            # bread_term = self.bread_term(self.Z, self.B, self.MU, self.Y)
            contrast_var_eta = self.full_covariance_contrast_variance(
                CB_flat, cov_beta, keepdims=True,
            )
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
            Cov_beta, _ = self.mass_univariate_fisher_covariance(self.Z, self.beta)
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
        p_vals = scipy.stats.norm.sf(z_stats) # shape: (_N, 1)
        print(p_vals.shape, z_stats.shape)
        print(np.min(p_vals), np.max(p_vals), np.count_nonzero(p_vals < 0.05), p_vals.shape)

        return p_vals, z_stats
    
    def meat_term(self, Z, B, MU, Y, batch_M=1000):
        if MU.shape != Y.shape:
            MU = MU.reshape(Y.shape) # shape: (_M, _N)
        def compute_meat():
            meat_term_1 = np.zeros((self._P * self._R, self._P * self._R)) # shape: (_P*_R, _P*_R)
            W = Y - MU
            BW = W.dot(B)    # shape (M, P)
            T = (Z[:, :, None] * BW[:, None, :]).reshape(self._M, self._P * self._R)  # shape (M, PR)
            meat_term = T.T.dot(T)   # shape (PR, PR)
            del W, BW, T
            gc.collect()
            return meat_term

        return self._load_or_compute_array(
            self.meat_term_filename,
            "meat_term",
            compute_meat,
            load_message="Loading precomputed meat term...",
        )
    
    def bread_term(self, Z, B, MU, Y, dtype=np.float64, chunk_rows=256, epsilon=1e-6):
        if MU.shape != Y.shape:
            MU = MU.reshape(Y.shape)
        def compute_bread():
            start_time = time.time()
            bread_term = self.compute_fisher_information(
                Z, B, MU, use_dask=False, cache_filename=None,
            )
            print(np.min(np.diag(bread_term)), np.mean(np.diag(bread_term)), np.max(np.diag(bread_term)), "bread term diag stats")
            print("Time taken for bread term computation:", time.time() - start_time)
            gc.collect()
            return bread_term

        return self._load_or_compute_array(
            self.bread_term_filename,
            "bread_term",
            compute_bread,
            load_message="Loading precomputed bread term...",
            compute_message="Computing bread term...",
        )
