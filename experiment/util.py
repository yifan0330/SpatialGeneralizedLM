"""Numerical utilities for spatial brain-lesion regression and inference.

The functions in this module are intentionally array-oriented: most operations
avoid materialising the full Kronecker design matrix and instead use equivalent
matrix products with ``Z``, ``B``, and reshaped coefficient arrays.
"""

import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm
from absl import logging
from scipy.optimize import minimize
import dask.array as da
from dask.diagnostics import ProgressBar
import nibabel as nib
import matplotlib.pyplot as plt


_DEFAULT_MEMMAP_PATH = "/tmp/mu.dat"
_GRADIENT_MEMMAP_PATH = "/tmp/X.dat"


def _to_dask_array(array, chunks):
    """Return ``array`` as a Dask array with the requested chunks."""
    if isinstance(array, da.Array):
        return array
    return da.from_array(array, chunks=chunks)


def _compute_with_progress(array):
    """Compute a Dask object while displaying a progress bar."""
    with ProgressBar():
        return array.compute()


def _raise_unknown_mode(mode):
    raise ValueError(f"Unknown mode = {mode}")


def _regularized_pinv(matrix, damping_factor):
    """Compute a pseudo-inverse after adding diagonal damping."""
    eye = np.eye(matrix.shape[0], dtype=matrix.dtype)
    return np.linalg.pinv(matrix + damping_factor * eye)


def _safe_exp(array):
    """Exponentiate with clipping to reduce overflow risk."""
    return np.exp(np.clip(array, -100, 100))


def create_lesion_mask(p_empirical, brain_mask, lesion_mask_filename, threshold=5e-4):
    """Create and save a binary lesion mask from empirical voxel probabilities."""
    brain_mask_data = brain_mask.get_fdata()
    full_volume = np.zeros(brain_mask_data.shape, dtype=np.float64)
    full_volume[brain_mask_data > 0] = p_empirical

    lesion_mask = (full_volume > threshold) & (brain_mask_data > 0)
    lesion_mask = lesion_mask.astype(np.uint8)
    logging.info(
        "Lesion mask summary: min=%d, max=%d, n_voxels=%d",
        int(lesion_mask.min()),
        int(lesion_mask.max()),
        int(lesion_mask.sum()),
    )

    lesion_mask_nii = nib.Nifti1Image(lesion_mask, affine=brain_mask.affine, header=brain_mask.header)
    nib.save(lesion_mask_nii, lesion_mask_filename)
    return

def preprocess_Z(simulated_dset, Z, polynomial_order):
    """Pre-process subject-level covariates for simulated or real datasets."""
    if simulated_dset:
        covariate_col = Z[:, 0]
        covariate_col = np.stack([covariate_col**i for i in range(1, polynomial_order+1)], axis=1)
        covariate_mean = np.mean(covariate_col, axis=0)
        covariate_std = np.std(covariate_col, axis=0)
        covariate_std = np.where(covariate_std == 0, 1.0, covariate_std)
        covariate_col = (covariate_col - covariate_mean) / covariate_std
        Z = np.concatenate([covariate_col, Z[:, 1:]], axis=1)
    else:
        Z = Z[:, 1:]  # remove the ID column
        SexF_hot_encoder = Z[:, 0].reshape(-1, 1)
        Age_col, Headsize_col, CVR_col = [Z[:, i:i+1] for i in [1, 2, 3]]
        Age_col = np.concatenate([Age_col**i for i in range(1, polynomial_order+1)], axis=1)
        scalar_covariates = np.concatenate((Age_col, Headsize_col, CVR_col), axis=1)
        scalar_covariates -= np.mean(scalar_covariates, axis=0)
        Z = np.concatenate([SexF_hot_encoder, scalar_covariates], axis=1)

    return Z

def kronecker_vector_product(Z, B, beta, use_dask=False, block_size=1000):
    """Efficiently compute ``kron(Z, B) @ beta`` without forming ``kron(Z, B)``.

    This uses the identity
    ``kron(Z, B) @ vec(beta) = vec(Z @ beta.reshape(R, P) @ B.T)``.

    Args:
        Z: Matrix of shape [_M, _R]
        B: Matrix of shape [_N, _P]
        beta: Matrix of shape [_R * _P, 1]

    Returns:
        Matrix of shape [_M * _N, 1]
    """
    _M, _R = Z.shape
    _N, _P = B.shape
    beta = beta.reshape((_R, _P))
    if use_dask:
        Z = _to_dask_array(Z, chunks=(block_size, _R))
        B = _to_dask_array(B, chunks=(block_size, _P))
        beta = _to_dask_array(beta, chunks=(_R, _P))
    return (Z @ beta @ B.T).reshape((_M * _N, 1))

def compute_gradient(Z, 
                     B, 
                     beta, 
                     Y, 
                     mode="approximate",
                     block_size=1000):
    """
    Z: [_M, _R]
    B: [_N, _P]
    beta: [_R * _P,]
    Y: [_M, _N] binary matrix
    """
    _M, _R = Z.shape
    _N, _P = B.shape
    if sparse.issparse(Y):
        Y = Y.tocsr()
    G = Y @ B  # [_M, _P]
    G = Z.T @ G  # [_R, _P]
    XTY = G.reshape((_R * _P, 1))  # [_R * _P, 1]
    if mode == "approximate":
        Z_bar = Z - Z.mean(axis=1, keepdims=True)
        eta_bar = compute_eta_mean(Z, B, beta)  # [_N, 1]
        exp_eta_bar = np.exp(eta_bar)
        XTmu = np.kron(Z.T.sum(axis=1, keepdims=True), B.T @ exp_eta_bar)  # [_R*_P, 1]
        B_bar = B * exp_eta_bar  # [_N, _P]
        XTmu += kronecker_vector_product(Z.T @ Z_bar, B.T @ B_bar, beta)  # [_R*_P, 1]
    elif mode == "offload":
        X = np.memmap(_GRADIENT_MEMMAP_PATH, dtype=np.float64, mode="w+", shape=(_M, _N))
        for j in tqdm(range(0, _N, block_size)):
            for i in range(0, _M, block_size):
                i_end = min(i + block_size, _M)
                j_end = min(j + block_size, _N)
                mu = np.exp(kronecker_vector_product(Z[i:i_end, :], B[j:j_end, :], beta))
                X[i:i_end, j:j_end] = mu.reshape((i_end-i, j_end-j))[:, :]
        X.flush()
        XTmu = np.zeros((_R, _P))
        for j in tqdm(range(0, _N, block_size)):
            for i in range(0, _M, block_size):
                i_end = min(i + block_size, _M)
                j_end = min(j + block_size, _N)
                XTmu += Z[i:i_end, :].T @ X[i:i_end, j:j_end] @ B[j:j_end, :]  # [_R, _P]
        XTmu = XTmu.reshape((_R * _P, 1))
    elif mode == "exact":
        eta = kronecker_vector_product(Z, B, beta)  # [_M*_N, 1]
        XTmu = kronecker_vector_product(Z.T, B.T, np.exp(eta))  # [_R*_P, 1]
    elif mode == "dask":
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        XTmu = kronecker_vector_product(Z.T, B.T, mu, use_dask=True, block_size=block_size)
        XTmu = _compute_with_progress(XTmu)
    else:
        _raise_unknown_mode(mode)
    return -(XTY - XTmu)

def compute_preconditioner(Z, 
                           B, 
                           beta=None,
                           mu_Z=None, 
                           mu_X=None, 
                           mode="approximate",
                           block_size=1000,
                           damping_factor=1e-4):
    if mode == "approximate":
        if mu_Z is None or mu_X is None:
            raise ValueError("mu_Z and mu_X are required for approximate preconditioning")
        ZTWZ = Z.T @ (Z * mu_Z)
        BTWB = B.T @ (B * mu_X)
        ZTWZ_inv = _regularized_pinv(ZTWZ, damping_factor)  # [_R, _R]
        BTWB_inv = _regularized_pinv(BTWB, damping_factor)  # [_P, _P]
        return np.kron(ZTWZ_inv, BTWB_inv)  # [_R*_P, _R*_P]
    elif mode == "dask":
        if beta is None:
            raise ValueError("beta is required for dask preconditioning")
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        XTmuX = efficient_kronT_diag_kron(Z, B, mu, use_dask=True, block_size=block_size)
        return _regularized_pinv(XTmuX, damping_factor)
    elif mode == "exact":
        if beta is None:
            raise ValueError("beta is required for exact preconditioning")
        mu = np.exp(kronecker_vector_product(Z, B, beta))
        XTmuX = efficient_kronT_diag_kron(Z, B, mu)
        return _regularized_pinv(XTmuX, damping_factor)
    else:
        _raise_unknown_mode(mode)

def compute_eta_mean(Z, 
                     B, 
                     beta):
    """
    Args:
      Z: [_M, _R]
      B: [_N, _P]
      beta: [_R * _P, 1]
    Returns:
      eta_bar: [_N, 1]
    """
    eta_bar = np.mean(Z, axis=0, keepdims=True)
    eta_bar = kronecker_vector_product(eta_bar, B, beta)
    return eta_bar

def SpatialGLM_compute_mu_mean(Z, 
                    B, 
                    beta,
                    mode="approximate",
                    block_size=100):
    """Compute voxel-wise mean and standard deviation of ``mu = exp(Z beta B.T)``."""
    _M, _R = Z.shape
    _N, _P = B.shape
    if mode == "exact":
        eta = kronecker_vector_product(Z, B, beta)
        mu = np.exp(eta)
        mu_mean = mu.reshape((_M, _N)).mean(axis=0)
        mu_std = mu.reshape((_M, _N)).std(axis=0)
        return mu_mean, mu_std
    elif mode == "dask":
        eta = kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size)
        mu = da.exp(eta)
        mu_mean = mu.reshape((_M, _N)).mean(axis=0)
        mu_std = mu.reshape((_M, _N)).std(axis=0)
        with ProgressBar():
            mu_mean, mu_std = da.compute(mu_mean, mu_std)
        return mu_mean, mu_std
    elif mode == "approximate":
        Z_bar = Z - Z.mean(axis=1, keepdims=True)
        eta_bar = compute_eta_mean(Z, B, beta)  # [_N, 1]
        exp_eta_bar = np.exp(eta_bar)  # [_N, 1]
        B_bar = B * exp_eta_bar
        Z_bar_mean = Z_bar.mean(axis=0, keepdims=True)
        mu_bar = exp_eta_bar + kronecker_vector_product(Z_bar_mean, B_bar, beta)
        return mu_bar, None
    elif mode == "offload":
        mu = np.memmap(_DEFAULT_MEMMAP_PATH, dtype=np.float64, mode="w+", shape=(_M, _N))
        for j in range(0, _N, block_size):
            for i in range(0, _M, block_size):
                i_end = min(i + block_size, _M)
                j_end = min(j + block_size, _N)
                mu[i:i_end, j:j_end] = np.exp(kronecker_vector_product(Z[i:i_end, :], B[j:j_end, :], beta)).reshape((i_end-i, j_end-j))[:, :]
        mu.flush()
        mu_mean = mu.mean(axis=0)
        mu_std = mu.std(axis=0)
        return mu_mean, mu_std
    else:
        _raise_unknown_mode(mode)

def SpatialGLM_compute_P_mean(Z, 
                   B, 
                   beta,
                   mode="approximate",
                   block_size=100):
    """Compute voxel-wise mean of ``P = mu * exp(-mu)``."""
    _M, _R = Z.shape
    _N, _P = B.shape
    if mode == "dask":
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        P = mu * da.exp(-mu)
        P_mean = P.reshape((_M, _N)).mean(axis=0)
        return _compute_with_progress(P_mean)
    if mode == "exact":
        mu = np.exp(kronecker_vector_product(Z, B, beta)).reshape((_M, _N))
        return np.mean(mu * np.exp(-mu), axis=0)
    if mode == "offload":
        mu = compute_mu(Z, B, beta, mode="offload", block_size=block_size)
        return np.mean(mu * np.exp(-mu), axis=0)
    if mode == "approximate":
        mu_mean, _ = SpatialGLM_compute_mu_mean(
            Z, B, beta, mode="approximate", block_size=block_size
        )
        return mu_mean * np.exp(-mu_mean)
    _raise_unknown_mode(mode)
    
def compute_mu(Z, 
               B, 
               beta,
               mode="dask",
               block_size=100):
    """Compute ``mu = exp(kron(Z, B) @ beta)`` using the requested backend."""
    _M, _R = Z.shape
    _N, _P = B.shape
    if mode == "exact":
        eta = kronecker_vector_product(Z, B, beta)
        mu = np.exp(eta)
        return mu
    elif mode == "dask":
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        return _compute_with_progress(mu)
    elif mode == "offload":
        mu = np.memmap(_DEFAULT_MEMMAP_PATH, dtype=np.float64, mode="w+", shape=(_M, _N))
        for j in range(0, _N, block_size):
            for i in range(0, _M, block_size):
                i_end = min(i + block_size, _M)
                j_end = min(j + block_size, _N)
                mu[i:i_end, j:j_end] = np.exp(kronecker_vector_product(Z[i:i_end, :], B[j:j_end, :], beta)).reshape((i_end-i, j_end-j))[:, :]
        mu.flush()
        return mu
    else:
        _raise_unknown_mode(mode)


# Backwards-compatible aliases used by older scripts/tests.
compute_mu_mean = SpatialGLM_compute_mu_mean
compute_P_mean = SpatialGLM_compute_P_mean

def log_poisson_likelihood(lam, Y, use_dask=False, block_size=1000):
    """Compute the log Poisson likelihood.
    Args:
        lam: Array of shape [n_samples, n_features]
        Y: Array of shape [n_samples, n_features]
    Returns:
        NLL: Negative log likelihood
    """
    if use_dask:
        lam = _to_dask_array(lam, chunks=(block_size,)).ravel()
        Y = _to_dask_array(Y, chunks=(block_size,)).ravel()
        lam = da.maximum(lam, np.finfo(float).tiny)
        return (Y * da.log(lam) - lam).mean()

    lam = np.asarray(lam).reshape(-1)
    lam = np.clip(lam, np.finfo(float).tiny, None)
    if sparse.issparse(Y):
        Y = Y.reshape((Y.shape[0] * Y.shape[1], 1))
        log_lam = np.log(lam).reshape((-1, 1))
        return (Y.multiply(log_lam) - lam.reshape((-1, 1))).mean()
    Y = np.asarray(Y).reshape(-1)
    return np.mean(Y * np.log(lam) - lam)

def compute_log_poisson_nll(Z, B, beta, Y, mode="exact", block_size=1000):
    _M, _R = Z.shape
    _N, _P = B.shape
    if mode == "exact":
        mu = np.exp(kronecker_vector_product(Z, B, beta))
        nll = -log_poisson_likelihood(mu, Y.reshape(-1, 1))
    elif mode == "dask":
        if not isinstance(Y, da.Array):
            Y = da.from_array(Y, chunks=(block_size, block_size))
        if not isinstance(Z, da.Array):
            Z = da.from_array(Z, chunks=(block_size, _R))
        if not isinstance(B, da.Array):
            B = da.from_array(B, chunks=(block_size, _P))
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        nll = -log_poisson_likelihood(mu, Y, use_dask=True, block_size=block_size)
        nll = nll.compute()
    else:
        _raise_unknown_mode(mode)
    return nll

def irls_log_glm(X, y, max_iter=50, tol=1e-10, compute_nll=False):
    """IRLS for Log Poisson GLM Regression.
    Args:
        X: Feature array array of shape [n_samples, n_features]
        y: Response array of shape [n_samples,]
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
    Returns:
        beta: Estimated coefficients
    """
    n_samples, n_features = X.shape
    logging.info("-" * 50)
    logging.info("IRLS for Log Poisson GLM")
    logging.info(f"n_samples: {n_samples}, n_features: {n_features}")
    logging.info("-" * 50)
    beta = np.zeros((n_features,))
    for iteration in range(max_iter):
        eta = X.dot(beta)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        XTmuX = X.T.dot(mu[:, None] * X)
        XTmuX = XTmuX + 1e-8 * np.eye(n_features)
        XTmuz = X.T.dot(mu * z)
        beta_new = np.linalg.solve(XTmuX, XTmuz)
        delta_beta = np.linalg.norm(beta_new - beta)
        beta = beta_new
        if compute_nll:
            nll = -log_poisson_likelihood(np.exp(X.dot(beta)), y)
            logging.info(f"--> Iteration: {iteration}, delta beta: {delta_beta}, NLL: {nll}")
        else:
            logging.info(f"--> Iteration: {iteration}, delta beta: {delta_beta}")
        if delta_beta < tol:
            logging.info(f"Converged in {iteration + 1} iterations.")
            break
    return beta

def fit_additive_log_glm(Z, B, Y, mode="approximate"):
    """Fit an additive log-Poisson approximation used for preconditioning."""
    _M, _R = Z.shape
    _N, _P = B.shape
    if mode == "exact":
        raise NotImplementedError
    if mode != "approximate":
        _raise_unknown_mode(mode)

    if sparse.issparse(Y):
        Y = Y.tocsr()
    beta, gamma = np.zeros((_P,)), np.zeros((_R,))
    Y_Z = np.asarray(Y.mean(axis=1)).reshape(-1)
    Y_B = np.asarray(Y.mean(axis=0)).reshape(-1)

    def objective(params):
        beta = params[:_P]
        gamma = params[_P:]
        B_beta = B @ beta
        Z_gamma = Z @ gamma
        likelihood = (
            (Y_B * B_beta).mean()
            + (Y_Z * Z_gamma).mean()
            - np.exp(B_beta).mean() * np.exp(Z_gamma).mean()
        )
        return -likelihood

    def jac(params):
        beta = params[:_P]
        gamma = params[_P:]
        B_beta = B @ beta
        Z_gamma = Z @ gamma
        exp_B_beta = np.exp(B_beta)
        exp_Z_gamma = np.exp(Z_gamma)
        d_beta = B.T @ Y_B / float(_N) - B.T @ exp_B_beta / float(_N) * exp_Z_gamma.mean()
        d_gamma = Z.T @ Y_Z / float(_M) - Z.T @ exp_Z_gamma / float(_M) * exp_B_beta.mean()
        return -np.concatenate([d_beta, d_gamma])

    res = minimize(
        fun=objective,
        jac=jac,
        x0=np.concatenate([beta, gamma]),
        method="L-BFGS-B",
        options={"disp": False},
    )
    return res.x[:_P], res.x[_P:]

def eigen_clip(M, min_val=0.1, max_val=10.0):
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals_clipped = np.clip(eigvals, min_val, max_val)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

def fit_multiplicative_log_glm(Z, B, Y, tol=1e-10, max_iter=100, 
                               alpha=1.0,
                               gradient_mode="approximate", 
                               preconditioner_mode="approximate",
                               nll_mode="dask",
                               block_size=1000,
                               compute_nll=False):
    """Fit Multiplicative Log GLM.
    Args:
        Z: Matrix of shape [_M, _R]
        B: Matrix of shape [_N, _P]
        Y: binary matrix of shape [_M, _N]
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
    Returns:
        beta: Estimated coefficients
    """
    _M, _R = Z.shape
    _N, _P = B.shape
    logging.info("-" * 50)
    logging.info("Multiplicative Log Poisson GLM")
    logging.info(f"n_subject: {_M}, n_voxel: {_N}, n_covariates: {_R}, n_basis: {_P}")
    logging.info("-" * 50)
    if preconditioner_mode == "approximate":
        beta_B, beta_Z = fit_additive_log_glm(Z, B, Y, mode=preconditioner_mode)
        mu_Z = np.exp(Z @ beta_Z)[:, None]
        mu_B = np.exp(B @ beta_B)[:, None]
    else:
        mu_Z, mu_B = None, None
    beta = np.zeros((_R * _P, 1))
    beta[-1, :] = np.log(max(float(np.mean(Y)), np.finfo(float).tiny))

    for iteration in range(max_iter):
        G = compute_gradient(Z, B, beta, Y, mode=gradient_mode, block_size=block_size)
        C = compute_preconditioner(Z, B, beta, mu_Z=mu_Z, mu_X=mu_B, 
                                   mode=preconditioner_mode, block_size=block_size, damping_factor=1e-4)
        beta_new = beta - alpha * C @ G
        if not np.isfinite(beta_new).all():
            raise FloatingPointError(
                "S-GLM regression produced non-finite beta values. "
                "Try reducing --sglm_alpha or using a smaller step size."
            )
        delta_beta = np.linalg.norm(beta_new - beta)
        rel_delta_beta = delta_beta / max(np.linalg.norm(beta), 1.0)
        beta = beta_new
        if compute_nll:
            nll = compute_log_poisson_nll(Z, B, beta, Y, mode=nll_mode, block_size=block_size)
            logging.info(f"--> Iteration: {iteration}, delta beta: {delta_beta}, rel delta beta: {rel_delta_beta}, NLL: {nll}")
            logging.info(f"--> Min beta: {np.min(beta)}, Max beta: {np.max(beta)}")
        else:
            logging.info(f"--> Iteration: {iteration}, delta beta: {delta_beta}, rel delta beta: {rel_delta_beta}")
        if delta_beta < tol or rel_delta_beta < tol:
            logging.info(f"Converged in {iteration + 1} iterations.")
            break
    else:
        logging.warning(
            f"S-GLM did not converge within max_iter={max_iter}; "
            f"last delta beta={delta_beta}, last rel delta beta={rel_delta_beta}. "
            "Increase --sglm_max_iter or reduce --sglm_tol if needed."
        )
    return beta

def fit_MUM_log_glm(Z, B, Y, marginal_dist, link_func,
                    tol, max_iter,
                    **kwargs):
    """Fit mass-univariate log/logic GLMs across all voxels at once."""
    _M, _R = Z.shape
    _N, _P = B.shape
    logging.info("-" * 50)
    logging.info("Mass Univariate Model with L-BFGS Optimization")
    logging.info(f"n_subject: {_M}, n_voxel: {_N}, n_covariates: {_R}")
    logging.info(f"marginal_dist: {marginal_dist}, link_func: {link_func}")
    logging.info("-" * 50)
    if link_func == "logit":
        inverse_link_func = lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))
    elif link_func == "log":
        inverse_link_func = _safe_exp
    else:
        raise ValueError(f"Unknown link function: {link_func}")

    beta_init = np.ones((_R * _N))

    def nll(beta_flat, marginal_dist, Y, Z, _N, _R):
        """Calculate the negative log-likelihood for all voxels."""
        beta = beta_flat.reshape(_R, _N)
        eta = Z @ beta
        mu = inverse_link_func(eta)

        if marginal_dist == "Poisson":
            return -(Y * eta - mu).sum()
        elif marginal_dist == "Bernoulli":
            weights = np.ones(Y.shape[1])
            mu = np.clip(mu, np.finfo(float).tiny, 1 - np.finfo(float).eps)
            return -(weights * np.log(mu) * Y + weights * np.log(1 - mu) * (1 - Y)).sum()
        raise ValueError(f"Unknown marginal distribution: {marginal_dist}")
    
    def gradient_poisson(beta_flat, marginal_dist, Y, Z, _N, _R):
        """Calculate the negative log-likelihood gradient for all voxels."""
        beta = beta_flat.reshape(_R, _N)
        eta = Z @ beta
        mu = inverse_link_func(eta)
        if marginal_dist == "Poisson":
            gradient = Z.T @ (mu - Y)
        elif marginal_dist == "Bernoulli":
            weights = np.ones(Y.shape[1])
            gradient = Z.T @ (weights * (mu - Y))
        else:
            raise ValueError(f"Unknown marginal distribution: {marginal_dist}")
        gradient_flat = gradient.ravel()
        if not np.all(np.isfinite(gradient_flat)):
            logging.warning("Gradient is NaN or Inf, optimization is unstable.")
        return gradient_flat

    logging.info("--- Starting L-BFGS Optimization for ALL VOXELS SIMULTANEOUSLY ---")
    logging.info(f"Total Parameters to Optimize: {_N*_R} (N_voxels * R_covariates)")

    optimization_args = (marginal_dist, Y, Z, _N, _R)

    result = minimize(
        fun=nll,
        x0=beta_init,
        args=optimization_args,
        method="L-BFGS-B",
        jac=gradient_poisson,
        tol=tol,
        options={"disp": True, "maxiter": max_iter},
    )

    logging.info("-" * 50)
    logging.info(f"Optimization Status: {result.message}")
    logging.info(f"Number of iterations: {result.nit}")
    logging.info(f"Final Total NLL (Minimum): {result.fun:.4f}")

    beta = result.x.reshape(_R, _N)
    
    return beta

def efficient_kronT_diag_kron(Z, B, d, use_dask=False, block_size=1000):
    """
    Efficiently computes kron(Z, B)^T @ diag(d) @ kron(Z, B)

    Args:
      Z : array of shape (_M, _R)
          The first matrix.
      B : array of shape (_N, _P)
          The second matrix.
      d : array of length (_M * _N)
          The diagonal entries of D, which will be reshaped to (_M, _N).
      use_dask : bool
          Whether to use dask for computation.
      block_size : int
          The block size for dask computation.

    Returns:
      result : array of shape (_R * _P, _R * _P)
    """
    _M, _R = Z.shape
    _N, _P = B.shape
    d_reshaped = d.reshape((_M, _N))
    if use_dask:
        Z = _to_dask_array(Z, chunks=(block_size, _R))
        B = _to_dask_array(B, chunks=(block_size, _P))
        d_reshaped = _to_dask_array(d_reshaped, chunks=(block_size, block_size))

        result = da.einsum("ij,jr,js,ik,il->klrs", d_reshaped, B, B, Z, Z, optimize="optimal")

        result = _compute_with_progress(result)
        return result.reshape(_R * _P, _R * _P)

    M = np.einsum("ij,jr,js->irs", d_reshaped, B, B, optimize=True)
    result_blocks = np.einsum("ik,il,irs->klrs", Z, Z, M, optimize=True)
    result = result_blocks.transpose(0, 2, 1, 3).reshape(_R * _P, _R * _P)
    return result

def robust_inverse(XTWX, eps=1e-8):
    """Compute a numerically robust symmetric pseudo-inverse."""
    XTWX = (XTWX + XTWX.T) / 2
    U, S, VT = np.linalg.svd(XTWX, full_matrices=False)
    eps = min(np.median(S), eps)
    M = (S > eps)
    S_inv = np.divide(1.0, S, out=np.zeros_like(S), where=M)
    U = ((U + VT.T) / 2) * M[None, :]
    XTWX_inv = U @ np.diag(S_inv) @ U.T
    return XTWX_inv

def robust_inverse_generalised(XTWX, Q, eps=1e-16):
    """Compute ``diag(Q @ pinv(XTWX) @ Q.T)`` robustly."""
    Q = Q.reshape(-1, XTWX.shape[0])
    if Q.shape[-1] != XTWX.shape[0]:
        raise ValueError("Mismatch in dimensions")
    XTWX = (XTWX + XTWX.T) / 2
    U, S, VT = np.linalg.svd(XTWX, full_matrices=False)
    M = (S > eps)
    S_inv = np.divide(1.0, S, out=np.zeros_like(S), where=M)
    U = ((U + VT.T) / 2) * M[None, :]
    QU = Q @ U
    diag_cov = np.sum(QU ** 2 * S_inv, axis=1)
    
    return diag_cov[:, None]

def eigenspectrum(A, save_path=None):
    """Plot and save the eigenspectrum of a symmetric matrix."""
    if save_path is None:
        raise ValueError("Path must be provided to save the eigenspectrum plot.")

    eigenvalues = np.linalg.eigvalsh(A)  # eigvalsh for symmetric/Hermitian matrices
    eigenvalues = np.sort(eigenvalues)[::-1]
    logging.info("Top eigenvalues: %s", eigenvalues[:10])

    plt.figure(figsize=(8, 4))
    plt.plot(eigenvalues, 'o-', markersize=3)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenspectrum')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()