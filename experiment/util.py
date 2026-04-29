import numpy as np
import scipy
from tqdm import tqdm
from absl import logging
from scipy.optimize import minimize, line_search
import dask.array as da
from dask.diagnostics import ProgressBar
import opt_einsum as oe
import nibabel as nib
import os
import matplotlib.pyplot as plt

def create_lesion_mask(p_empirical, brain_mask, lesion_mask_filename, threshold=5e-4):
    # Load brain mask
    brain_mask_data = brain_mask.get_fdata()
    full_volume = np.zeros(brain_mask_data.shape)  # Create an empty volume
    full_volume[brain_mask_data > 0] = p_empirical  # Fill only where mask is nonzero
    p_empirical_map = nib.Nifti1Image(full_volume, affine=brain_mask.affine, header=brain_mask.header)

    # Load probability map
    p_empirical_map = p_empirical_map.get_fdata()

    # Create binary lesion mask: 1 where probability > threshold & inside the brain mask
    lesion_mask = (p_empirical_map > threshold) & (brain_mask_data == 1)

    # Convert to integer format (0 and 1)
    lesion_mask = lesion_mask.astype(np.uint8)
    print(np.min(lesion_mask), np.max(lesion_mask), np.sum(lesion_mask))

    # Create a new NIfTI image using the original brain mask affine & header
    lesion_mask_nii = nib.Nifti1Image(lesion_mask, affine=brain_mask.affine, header=brain_mask.header)

    # Save to disk
    nib.save(lesion_mask_nii, lesion_mask_filename)

    return 

def preprocess_Z(simulated_dset, Z, polynomial_order):
    if simulated_dset:
        covariate_col = Z[:, 0]
        covariate_col = np.stack([covariate_col**i for i in range(1, polynomial_order+1)], axis=1)
        covariate_col = (covariate_col - np.mean(covariate_col, axis=0)) / np.std(covariate_col, axis=0)
        # Z = covariate_col
        Z = np.concatenate([covariate_col, Z[:, 1:]], axis=1)
    else:
        Z = Z[:,1:] # remove the ID column
        SexF_hot_encoder = Z[:, 0].reshape(-1, 1)
        # SexF_hot_encoder = np.stack([(Z[:,0] == i).astype(int) for i in np.unique(Z[:,0])], axis=1)
        Age_col, Headsize_col, CVR_col = [Z[:, i:i+1] for i in [1, 2, 3]]
        # Polynomial order (Only for Age)
        Age_col = np.concatenate([Age_col**i for i in range(1, polynomial_order+1)], axis=1)
        scalar_covariates = np.concatenate((Age_col, Headsize_col, CVR_col), axis=1)
        scalar_covariates -= np.mean(scalar_covariates, axis=0)
        # scalar_covariates /= np.std(scalar_covariates, axis=0)
        Z = np.concatenate([SexF_hot_encoder, scalar_covariates], axis=1)
        # [Sex, Age, Headsize, CVR]

    return Z

def kronecker_vector_product(Z, B, beta, use_dask=False, block_size=1000):
    """ Efficient implementation of Kron(Z, B)beta.

    1. Reshape X to [_R, _P]
    2. Compute quadratic form of Z @ beta @ B.T

    Args:
        Z: Matrix of shape [_M, _R]
        B: Matrix of shape [_N, _P]
        beta: Matrix of shape [_R * _P, 1]
    Returns:
        Matrix of shape [_M * _N, 1]
    """
    _M, _R =  Z.shape
    _N, _P =  B.shape
    beta = np.reshape(beta, (_R, _P))
    if use_dask:
        if not isinstance(Z, da.Array):
            Z = da.from_array(Z, chunks=(block_size, _R))
        if not isinstance(B, da.Array):
            B = da.from_array(B, chunks=(block_size, _P))
        if not isinstance(beta, da.Array):
            beta = da.from_array(beta, chunks=(_R, _P))
        ret = (Z @ beta @ B.T).reshape((_M*_N, 1))
        return ret
    return np.einsum('...mr,...rp,...np->...mn', Z, beta, B).reshape((_M*_N, 1))

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
    if scipy.sparse.issparse(Y):
        Y = Y.tocsr()
    G = Y @ B # [_M, _P]
    G = Z.T @ G # [_R, _P]
    XTY = G.reshape((_R * _P, 1)) # [_R * _P, 1]
    if mode == "approximate":
        Z_bar = Z - Z.mean(axis=1, keepdims=True)
        eta_bar = compute_eta_mean(Z, B, beta) # [_N, 1]
        exp_eta_bar = np.exp(eta_bar)
        XTmu = np.kron(Z.T.sum(axis=1, keepdims=True), B.T @ exp_eta_bar) # [_R*_P, 1]
        B_bar = B * exp_eta_bar # [_N, _P]
        XTmu += kronecker_vector_product(Z.T @ Z_bar, B.T @ B_bar, beta) # [_R*_P, 1]
    elif mode == "offload":
        X = np.memmap("/tmp/X.dat", dtype=np.float64, mode="w+", shape=(_M, _N))
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
                XTmu += Z[i:i_end, :].T @ X[i:i_end, j:j_end] @ B[j:j_end, :] # [_R, _P]
        XTmu = XTmu.reshape((_R*_P, 1))
    elif mode == "exact":
        eta = kronecker_vector_product(Z, B, beta) # [_M*_N, 1]
        XTmu = kronecker_vector_product(Z.T, B.T, np.exp(eta)) # [_R*_P, 1]
    elif mode == "dask":
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        # mu = da.maximum(mu, epsilon)
        XTmu = kronecker_vector_product(Z.T, B.T, mu, use_dask=True, block_size=block_size)
        with ProgressBar():
            XTmu = XTmu.compute()
    else:
        raise ValueError("Unknown mode = {}".format(mode))
    return -(XTY - XTmu)

def compute_preconditioner(Z, 
                           B, 
                           beta=None,
                           mu_Z=None, 
                           mu_X=None, 
                           mode="approximate",
                           block_size=1000,
                           damping_factor=1e-4):
    _M, _R = Z.shape
    _N, _P = B.shape
    if mode == "approximate":
        assert mu_Z is not None and mu_X is not None
        ZTWZ = Z.T @ (Z * mu_Z)
        BTWB = B.T @ (B * mu_X)
        ZTWZ_inv = np.linalg.pinv(ZTWZ + damping_factor * np.eye(ZTWZ.shape[0])) # [_R, _R]
        BTWB_inv = np.linalg.pinv(BTWB + damping_factor * np.eye(BTWB.shape[0])) # [_P, _P]
        XTWX_inv = np.kron(ZTWZ_inv, BTWB_inv)  # [_R*_P, _R*_P]
        return XTWX_inv
    elif mode == "dask":
        assert beta is not None
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        XTmuX = efficient_kronT_diag_kron(Z, B, mu, use_dask=True, block_size=block_size)
        return np.linalg.pinv(XTmuX + damping_factor * np.eye(XTmuX.shape[0]))
    elif mode == "exact":
        assert beta is not None
        mu = np.exp(kronecker_vector_product(Z, B, beta))
        ZB = np.kron(Z, B) * (mu**0.5) # [_M*_N, _R*_P]
        XTmuX = ZB.T @ ZB # [_R*_P, _R*_P]
        return np.linalg.pinv(XTmuX + damping_factor * np.eye(XTmuX.shape[0]))
    else:
      raise ValueError("Unknown mode = {}".format(mode))

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
            mu_mean = mu_mean.compute()
            mu_std = mu_std.compute()
        return mu_mean, mu_std
    elif mode == "approximate":
        Z_bar = Z - Z.mean(axis=1, keepdims=True)
        eta_bar = compute_eta_mean(Z, B, beta) # [_N, 1]
        exp_eta_bar = np.exp(eta_bar) # [_N, 1]
        B_bar = B * exp_eta_bar
        Z_bar_mean = Z_bar.mean(axis=0, keepdims=True)
        mu_bar = exp_eta_bar + kronecker_vector_product(Z_bar_mean, B_bar, beta)
        return mu_bar, None
    elif mode == "offload":
        mu = np.memmap("/tmp/mu.dat", dtype=np.float64, mode="w+", shape=(_M, _N))
        for j in range(0, _N, block_size):
            for i in range(0, _M, block_size):
                i_end = min(i + block_size, _M)
                j_end = min(j + block_size, _N)
                mu[i:i_end, j:j_end] = np.exp(kronecker_vector_product(Z[i:i_end, :], B[j:j_end, :], beta)).reshape((i_end-i, j_end-j))[:, :]
        mu.flush()
        mu_mean = mu.mean(axis=0)
        mu_std = mu.std(axis=0)
        return mu_mean, mu_std

def SpatialGLM_compute_P_mean(Z, 
                   B, 
                   beta,
                   mode="approximate",
                   block_size=100):
    _M, _R = Z.shape
    _N, _P = B.shape
    print(_M, _N, _M*_N)
    if mode == "dask":
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        P = mu*da.exp(-mu)
        P_mean = P.reshape((_M, _N)).mean(axis=0)
        with ProgressBar():
            P_mean = P_mean.compute()
        return P_mean
    
def compute_mu(Z, 
               B, 
               beta,
               mode="dask",
               block_size=100):
    _M, _R = Z.shape
    _N, _P = B.shape
    if mode == "exact":
        eta = kronecker_vector_product(Z, B, beta)
        mu = np.exp(eta)
        return mu
    elif mode == "dask":
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        with ProgressBar():
            mu = mu.compute()
        return mu
    elif mode == "offload":
        mu = np.memmap("/tmp/mu.dat", dtype=np.float64, mode="w+", shape=(_M, _N))
        for j in range(0, _N, block_size):
            for i in range(0, _M, block_size):
                i_end = min(i + block_size, _M)
                j_end = min(j + block_size, _N)
                mu[i:i_end, j:j_end] = np.exp(kronecker_vector_product(Z[i:i_end, :], B[j:j_end, :], beta)).reshape((i_end-i, j_end-j))[:, :]
        mu.flush()
        return mu

def log_poisson_likelihood(lam, Y, use_dask=False, block_size=1000):
    """Compute the log Poisson likelihood.
    Args:
        lam: Array of shape [n_samples, n_features]
        Y: Array of shape [n_samples, n_features]
    Returns:
        NLL: Negative log likelihood
    """
    if use_dask:
        if not isinstance(lam, da.Array):
            lam = da.from_array(lam, chunks=(block_size,))
        if not isinstance(Y, da.Array):
            Y = da.from_array(Y, chunks=(block_size,))
        lam, Y = lam.ravel(), Y.ravel()
        return (da.multiply(Y, da.log(lam)) - lam).mean()
    return (Y.multiply(np.log(lam)) - lam).mean()

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
    _M, _R = Z.shape
    _N, _P = B.shape
    if scipy.sparse.issparse(Y):
        Y = Y.tocsr()
    if mode == "approximate":
        beta, gamma = np.zeros((_P,)), np.zeros((_R,))
        Y_Z = np.array(Y.mean(axis=1)).reshape(-1,)
        Y_B = np.array(Y.mean(axis=0)).reshape(-1,)
        def objective(params):
            beta = params[:_P]
            gamma = params[_P:]
            B_beta = B @ beta
            Z_gamma = Z @ gamma
            l = (Y_B * B_beta).mean() + (Y_Z * Z_gamma).mean() - np.exp(B_beta).mean() * np.exp(Z_gamma).mean()
            return -l
        def jac(params):
            beta = params[:_P]
            gamma = params[_P:]
            B_beta = B @ beta
            Z_gamma = Z @ gamma
            d_beta = B.T @ Y_B / float(_N) - B.T @ np.exp(B_beta) / float(_N) * np.exp(Z_gamma).mean()
            d_gamma = Z.T @ Y_Z / float(_M) - Z.T @ np.exp(Z_gamma) / float(_M) * np.exp(B_beta).mean()
            return -np.concatenate([d_beta, d_gamma])
        res = minimize(fun=objective, jac=jac, x0=np.concatenate([beta, gamma]), 
                    method="L-BFGS-B", options={"disp": True})
        beta = res.x[:_P]
        gamma = res.x[_P:]
        return beta, gamma
    elif mode == "exact":
        raise NotImplementedError

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
        mu_B = np.exp(B @ beta_B)[: , None]
    else:
        mu_Z, mu_B = None, None
    beta = np.zeros((_R * _P, 1))
    beta[-1, :] = np.log(np.mean(Y))

    for iteration in range(max_iter):
        G = compute_gradient(Z, B, beta, Y, mode=gradient_mode, block_size=block_size)
        C = compute_preconditioner(Z, B, beta, mu_Z=mu_Z, mu_X=mu_B, 
                                   mode=preconditioner_mode, block_size=block_size, damping_factor=1e-4)
        # C = eigen_clip(C, min_val=-1e4, max_val=1e4)
        beta_new = beta - alpha * C @ G
        # if iteration % 5 == 0:
        #     np.savez(f"{checkpoint_path}/iter_{iteration}.npy", beta=beta_new, G=G, C=C)
        delta_beta = np.linalg.norm(beta_new - beta)
        beta = beta_new
        if compute_nll:
            nll = compute_log_poisson_nll(Z, B, beta, Y, mode=nll_mode, block_size=block_size)
            logging.info(f"--> Iteration: {iteration}, delta beta: {delta_beta}, NLL: {nll}")
            logging.info(f"--> Min beta: {np.min(beta)}, Max beta: {np.max(beta)}")
        else:
            logging.info(f"--> Iteration: {iteration}, delta beta: {delta_beta}")
        if delta_beta < tol:
            logging.info(f"Converged in {iteration + 1} iterations.")
            break
    return beta

def fit_MUM_log_glm(Z, B, Y, marginal_dist, link_func,
                    tol, max_iter,
                    **kwargs):
    _M, _R = Z.shape
    _N, _P = B.shape
    logging.info("-" * 50)
    logging.info("Mass Univariate Model with L-BFGS Optimization")
    logging.info(f"n_subject: {_M}, n_voxel: {_N}, n_covariates: {_R}")
    logging.info(f"marginal_dist: {marginal_dist}, link_func: {link_func}")
    logging.info("-" * 50)
    # Define link function and its inverse
    if link_func == "logit":
        inverse_link_func = lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))
    elif link_func == "log":
        inverse_link_func = lambda x: np.exp(np.clip(x, -100, 100))
    else:
        raise ValueError("Unknown link function: {}".format(link_func))
    # initialize beta
    beta_init = x0 = np.ones((_R * _N)) #np.random.randn(_R * _N) * 1e-4

    def nll(beta_flat, marginal_dist, Y, Z, _N, _R):
        """Calculates the Negative Log-Likelihood (NLL) for a single voxel (j)."""
        # Linear Predictor: eta = Z * beta
        beta = beta_flat.reshape(_R, _N)
        eta = Z @ beta
        # Expected Mean (mu) using inverse link: mu = exp(eta)
        mu = inverse_link_func(eta)
        
        # Poisson Log-Likelihood: L = Sum(Y_i * log(mu_i) - mu_i - log(Y_i!))
        # Note: log(Y_i!) term can be ignored during optimization as it's a constant.
        
        # NLL = -Sum(Y_i * log(mu_i) - mu_i)
        if marginal_dist == "Poisson":
            nll = -(Y * eta - mu).sum()
        elif marginal_dist == "Bernoulli":
            weights = np.ones(Y.shape[1])
            nll = -(weights*np.log(mu) * Y + weights*np.log(1 - mu) * (1 - Y)).sum()
        else:
            raise ValueError("Unknown marginal distribution: {}".format(marginal_dist))
        return nll
    
    def gradient_poisson(beta_flat, marginal_dist, Y, Z, _N, _R):
        """
        Calculates the gradient of the NLL for a single voxel (j) w.r.t. beta.
        Gradient: grad(L) = Z^T * (Y - mu)
        Gradient(NLL) = -Z^T * (Y - mu) = Z^T * (mu - Y)
        """
        # Linear Predictor: eta = Z * beta
        beta = beta_flat.reshape(_R, _N)
        eta = Z @ beta

        # Expected Mean (mu)
        mu = inverse_link_func(eta)
        if marginal_dist == "Poisson":
            # Derivative of NLL w.r.t. beta (standard GLM result: Z^T * (mu - Y))
            gradient = (Z.T @ (mu - Y))
        elif marginal_dist == "Bernoulli":
            weights = np.ones(Y.shape[1])
            gradient = (Z.T @ (weights * (mu - Y)))
        else:
            raise ValueError("Unknown marginal distribution: {}".format(marginal_dist))
        gradient_flat = gradient.ravel()
        # Optional: Check for NaN/Inf in gradient
        if not np.all(np.isfinite(gradient_flat)):
            logging.warning("Gradient is NaN or Inf, optimization is unstable.")
        return gradient_flat

    logging.info("--- Starting L-BFGS Optimization for ALL VOXELS SIMULTANEOUSLY ---")
    logging.info(f"Total Parameters to Optimize: {_N*_R} (N_voxels * R_covariates)")

    # Arguments to pass to fun and jac functions
    optimization_args = (marginal_dist, Y, Z, _N, _R)

    # The minimize function performs the L-BFGS optimization.
    result = minimize(
        fun=nll,
        x0=beta_init,
        args=optimization_args,
        method='L-BFGS-B', # Bounded version, but works for unbounded if bounds=None
        jac=gradient_poisson,
        tol=tol,
        options={'disp': True, 'maxiter': max_iter}
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
    if use_dask:
        if not isinstance(Z, da.Array):
            Z = da.from_array(Z, chunks=(block_size, _R))
        if not isinstance(B, da.Array):
            B = da.from_array(B, chunks=(block_size, _P))
        d_reshaped = d.reshape((_M, _N))
        if not isinstance(d, da.Array):
            d = da.from_array(d, chunks=(block_size, block_size))

        result = da.einsum('ij,jr,js,ik,il->klrs', d_reshaped, B, B, Z, Z, optimize='optimal')

        with ProgressBar():
            result = result.compute()
        result = result.reshape(_R * _P, _R * _P)
        return result
    d_reshaped = d.reshape((_M, _N))
    M = np.einsum('ij,jr,js->irs', d_reshaped, B, B)  
    result_blocks = np.einsum('ik,il,irs->klrs', Z, Z, M)
    result = result_blocks.transpose(0, 2, 1, 3).reshape(_R * _P, _R * _P)
    return result

def robust_inverse(XTWX, eps=1e-8):
    XTWX = (XTWX + XTWX.T) / 2
    U, S, VT = np.linalg.svd(XTWX, full_matrices=False)
    # choose the threshold that at least 50% eigenvalues are kept
    eps = min(np.median(S), eps)
    M = (S > eps)
    # S_inv = (S ** -1) * M
    # XTWX_inv = VT.T @ np.diag(S_inv) @ U.T
    S_inv = S ** -1
    U = ((U + VT.T) / 2) * M[None, :]
    XTWX_inv = U @ np.diag(S_inv) @ U.T
    return XTWX_inv

def robust_inverse_generalised(XTWX, Q, eps=1e-16):
    Q = Q.reshape(-1, XTWX.shape[0])
    if Q.shape[-1] != XTWX.shape[0]:
        raise ValueError("Mismatch in dimensions")
    XTWX = (XTWX + XTWX.T) / 2
    U, S, VT = np.linalg.svd(XTWX, full_matrices=False)
    M = (S > eps)
    S_inv = S ** -1 # (S ** -1) * M + np.min(S)**(-1) * (1-M)
    U = ((U + VT.T) / 2) * M[None, :]
    QU = Q @ U
    diag_cov = np.sum(QU ** 2 * S_inv, axis=1)
    
    return diag_cov[:, None]

def eigenspectrum(A, save_path=None):
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(A)  # eigvalsh for symmetric/Hermitian matrices

    # Sort in decreasing order
    eigenvalues = np.sort(eigenvalues)[::-1]
    print(eigenvalues)

    # Plot the eigenspectrum
    plt.figure(figsize=(8, 4))
    plt.plot(eigenvalues, 'o-', markersize=3)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenspectrum')
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        raise ValueError("Path must be provided to save the eigenspectrum plot.")