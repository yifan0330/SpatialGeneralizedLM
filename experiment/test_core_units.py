"""Unit tests for core model, regression, inference, and utility routines.

Run from the repository root with:

    python -m unittest experiment/test_core_units.py

or from ``experiment/`` with:

    python -m unittest test_core_units.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sparse
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from inference import BrainInference, _BaseInferenceBackend  # noqa: E402
from model import MassUnivariateRegression, SpatialBrainLesionModel  # noqa: E402
from regression import (  # noqa: E402
    BrainRegression_Approximate,
    BrainRegression_full,
)
from util import (  # noqa: E402
    SpatialGLM_compute_P_mean,
    SpatialGLM_compute_mu_mean,
    compute_gradient,
    compute_mu,
    compute_preconditioner,
    efficient_kronT_diag_kron,
    kronecker_vector_product,
    log_poisson_likelihood,
    robust_inverse,
    robust_inverse_generalised,
)


def _object_array(payload):
    """Wrap a dict in a 0-d object array, matching saved simulation files."""
    array = np.empty((), dtype=object)
    array[()] = payload
    return array


class TestModelDefinitions(unittest.TestCase):
    """Tests for the PyTorch model definitions in model.py."""

    def setUp(self):
        torch.manual_seed(123)
        self.dtype = torch.float64

    def test_spatial_model_single_group_forward_loss_and_gradient(self):
        X = torch.randn(5, 3, dtype=self.dtype)
        Z = torch.randn(4, 2, dtype=self.dtype)
        Y = torch.bernoulli(torch.full((4, 5), 0.35, dtype=self.dtype))

        model = SpatialBrainLesionModel(
            n_covariates=2,
            n_auxiliary=1,
            n_bases=3,
            link_func="logit",
            marginal_dist="Bernoulli",
            dtype=self.dtype,
        )

        P = model(X, Y, Z)
        loss = model.get_loss(P, Y, Z)
        loss.backward()

        self.assertEqual(P.shape, Y.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.beta.grad)

    def test_spatial_model_group_specific_betas(self):
        X = {
            "Group_1": torch.randn(5, 3, dtype=self.dtype),
            "Group_2": torch.randn(5, 3, dtype=self.dtype),
        }
        Z = {
            "Group_1": torch.randn(4, 2, dtype=self.dtype),
            "Group_2": torch.randn(6, 2, dtype=self.dtype),
        }
        Y = {
            "Group_1": torch.bernoulli(torch.full((4, 5), 0.3, dtype=self.dtype)),
            "Group_2": torch.bernoulli(torch.full((6, 5), 0.3, dtype=self.dtype)),
        }

        model = SpatialBrainLesionModel(
            n_covariates=2,
            n_auxiliary=1,
            n_bases=3,
            group_names=["Group_1", "Group_2"],
            dtype=self.dtype,
        )

        P = model(X, Y, Z)
        loss = model.get_loss(P, Y, Z)
        loss.backward()

        self.assertEqual(set(P), {"Group_1", "Group_2"})
        self.assertIsNone(model.beta)
        self.assertTrue(all(model.betas[group].grad is not None for group in model.betas))

    def test_mass_univariate_forward_loss_and_firth_penalty(self):
        X = torch.randn(5, 3, dtype=self.dtype)
        Z = torch.randn(8, 2, dtype=self.dtype)
        Y = torch.bernoulli(torch.full((8, 5), 0.25, dtype=self.dtype))

        model = MassUnivariateRegression(
            n_covariates=2,
            n_auxiliary=1,
            n_voxels=5,
            firth_penalty=True,
            dtype=self.dtype,
        )

        P = model(X, Y, Z)
        loss = model.get_loss(P, Y, Z)
        loss.backward()

        self.assertEqual(P.shape, Y.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.beta.grad)

    def test_static_spatial_nll_supports_hessian(self):
        X = torch.randn(5, 3, dtype=self.dtype)
        Z = torch.randn(4, 2, dtype=self.dtype)
        Y = torch.bernoulli(torch.full((4, 5), 0.35, dtype=self.dtype))
        beta = torch.randn(3, 2, dtype=self.dtype, requires_grad=True)

        nll = SpatialBrainLesionModel._neg_log_likelihood(
            "Bernoulli", "logit", [], X, Y, Z, beta
        )
        hessian = torch.autograd.functional.hessian(
            lambda b: SpatialBrainLesionModel._neg_log_likelihood(
                "Bernoulli", "logit", [], X, Y, Z, b
            ),
            beta,
        )

        self.assertTrue(torch.isfinite(nll))
        self.assertEqual(hessian.shape, (3, 2, 3, 2))

    def test_spatial_model_log_link_poisson_loss_is_finite(self):
        X = torch.randn(5, 3, dtype=self.dtype)
        Z = torch.randn(4, 2, dtype=self.dtype)
        Y = torch.poisson(torch.full((4, 5), 0.5, dtype=self.dtype))

        model = SpatialBrainLesionModel(
            n_covariates=2,
            n_auxiliary=1,
            n_bases=3,
            link_func="log",
            marginal_dist="Poisson",
            dtype=self.dtype,
        )

        P = model(X, Y, Z)
        loss = model.get_loss(P, Y, Z)

        self.assertTrue(torch.all(P > 0))
        self.assertTrue(torch.isfinite(loss))

    def test_arctanh_link_maps_to_unit_interval(self):
        X = torch.randn(5, 3, dtype=self.dtype)
        Z = torch.randn(4, 2, dtype=self.dtype)
        Y = torch.bernoulli(torch.full((4, 5), 0.35, dtype=self.dtype))

        model = SpatialBrainLesionModel(
            n_covariates=2,
            n_auxiliary=1,
            n_bases=3,
            link_func="arctanh",
            marginal_dist="Bernoulli",
            dtype=self.dtype,
        )

        P = model(X, Y, Z)

        self.assertTrue(torch.all(P >= 0))
        self.assertTrue(torch.all(P <= 1))

    def test_static_mass_univariate_nll_is_differentiable(self):
        Z = torch.randn(6, 2, dtype=self.dtype)
        Y = torch.bernoulli(torch.full((6, 4), 0.25, dtype=self.dtype))
        beta_param = torch.randn(4, dtype=self.dtype, requires_grad=True)
        beta_other = torch.randn(2, 4, dtype=self.dtype)
        beta_other[1] = 0.0

        nll = MassUnivariateRegression._neg_log_likelihood(
            "Bernoulli", "logit", [], None, Y, Z, beta_param, beta_other
        )
        nll.backward()

        self.assertTrue(torch.isfinite(nll))
        self.assertIsNotNone(beta_param.grad)
        self.assertEqual(beta_param.grad.shape, beta_param.shape)


class TestRegressionBackends(unittest.TestCase):
    """Tests for data loading and regression orchestration."""

    def setUp(self):
        self.flat_data = {
            "X_spatial": np.array([[1.0], [2.0], [3.0]]),
            "Z": np.array([[0.1], [0.2], [0.3], [0.4]]),
            "Y": np.ones((4, 3)),
        }
        self.group_data = {
            "Group_1": _object_array(
                {
                    "X_spatial": np.array([[1.0], [2.0], [3.0]]),
                    "Z": np.array([[0.1], [0.2], [0.3], [0.4]]),
                    "Y": np.ones((4, 3)),
                }
            ),
            "Group_2": _object_array(
                {
                    "X_spatial": np.array([[1.0], [2.0], [3.0]]),
                    "Z": np.array([[0.5], [0.6], [0.7]]),
                    "Y": np.ones((3, 3)),
                }
            ),
        }

    def test_approximate_loads_flat_data_with_scaled_intercepts(self):
        regression = BrainRegression_Approximate(simulated_dset=False)
        regression.load_data(self.flat_data, model="SpatialBrainLesion")

        self.assertEqual(regression.B.shape, (3, 2))
        self.assertEqual(regression.Z.shape, (4, 2))
        self.assertTrue(np.allclose(regression.B[:, -1], 1.0))
        self.assertTrue(np.allclose(regression.Z[:, -1], 1.0))

    def test_approximate_grouped_spatial_regression_dispatches_per_group(self):
        regression = BrainRegression_Approximate(simulated_dset=True)
        regression.load_data(self.group_data, model="SpatialBrainLesion")
        fake_beta = np.zeros((regression._R * regression._P, 1))

        with patch("regression.fit_multiplicative_log_glm", return_value=fake_beta) as mocked_fit:
            beta = regression.run_regression(
                model="SpatialBrainLesion",
                marginal_dist="Poisson",
                link_func="log",
                max_iter=2,
            )

        self.assertEqual(set(beta), {"Group_1", "Group_2"})
        self.assertEqual(mocked_fit.call_count, 2)

    def test_full_regression_initialises_model_from_loaded_group_data(self):
        regression = BrainRegression_full(dtype=torch.float64)
        regression.load_data(self.group_data)
        regression.init_model(
            "SpatialBrainLesion",
            n_auxiliary=1,
            std_auxiliary=1.0,
            n_samples=5,
            regression_terms=None,
            link_func="logit",
            marginal_dist="Bernoulli",
            firth_penalty=False,
        )

        self.assertEqual(regression.group_names, ["Group_1", "Group_2"])
        self.assertIsNotNone(regression.model.betas)
        self.assertEqual(regression.model.n_bases, regression.n_bases_scalar)

    def test_full_regression_initialises_mass_univariate_model(self):
        regression = BrainRegression_full(dtype=torch.float64)
        regression.load_data(self.group_data)
        regression.init_model(
            "MassUnivariateRegression",
            n_auxiliary=1,
            std_auxiliary=1.0,
            n_samples=5,
            regression_terms=None,
            link_func="logit",
            marginal_dist="Bernoulli",
            firth_penalty=True,
        )

        self.assertIsInstance(regression.model, MassUnivariateRegression)
        self.assertTrue(regression.model.firth_penalty)
        self.assertEqual(regression.model.n_voxels, regression.n_voxels_scalar)

    def test_approximate_mass_univariate_regression_dispatches_solver(self):
        regression = BrainRegression_Approximate(simulated_dset=False)
        regression.load_data(self.flat_data, model="MassUnivariateRegression")
        fake_beta = np.zeros((regression._R, regression._N))

        with patch("regression.fit_MUM_log_glm", return_value=fake_beta) as mocked_fit:
            beta = regression.run_regression(
                model="MassUnivariateRegression",
                marginal_dist="Poisson",
                link_func="log",
                max_iter=2,
            )

        self.assertTrue(np.array_equal(beta, fake_beta))
        mocked_fit.assert_called_once()

    def test_regression_goodness_of_fit_for_mass_univariate_model(self):
        regression = BrainRegression_Approximate(simulated_dset=False)
        regression.load_data(self.flat_data, model="MassUnivariateRegression")
        beta = np.zeros((regression._R, regression._N))

        MU_mean, MU_std, P_mean = regression.goodness_of_fit(
            beta=beta, model="MassUnivariateRegression"
        )

        self.assertIsNone(MU_mean)
        self.assertIsNone(MU_std)
        self.assertTrue(np.allclose(P_mean, np.exp(-1.0)))

    def test_grouped_goodness_of_fit_dispatches_per_group(self):
        regression = BrainRegression_Approximate(simulated_dset=True)
        regression.load_data(self.group_data, model="SpatialBrainLesion")
        beta = {group: np.zeros((regression._R * regression._P, 1)) for group in regression.group_names}

        with patch.object(
            BrainRegression_Approximate,
            "_compute_spatial_gof",
            return_value=(np.ones(3), np.zeros(3), np.full(3, 0.5)),
        ) as mocked_gof:
            MU_mean, MU_std, P_mean = regression.goodness_of_fit(
                beta=beta, model="SpatialBrainLesion"
            )

        self.assertEqual(set(MU_mean), {"Group_1", "Group_2"})
        self.assertEqual(set(MU_std), {"Group_1", "Group_2"})
        self.assertEqual(set(P_mean), {"Group_1", "Group_2"})
        self.assertEqual(mocked_gof.call_count, 2)


class TestInferenceFacadeAndHelpers(unittest.TestCase):
    """Tests for the unified inference facade and shared helpers."""

    def _backend(self, link_func="log"):
        return _BaseInferenceBackend(
            model="SpatialBrainLesion",
            marginal_dist="Poisson",
            link_func=link_func,
            regression_terms=None,
        )

    def test_invalid_inference_type_raises(self):
        with self.assertRaises(ValueError):
            BrainInference(
                model="SpatialBrainLesion",
                marginal_dist="Poisson",
                link_func="log",
                regression_terms=None,
                inference_type="missing",
            )

    def test_base_contrast_helpers_normalise_and_validate(self):
        backend = self._backend()

        contrast = backend._set_contrast([[2.0, -2.0]], expected_width=2, contrast_name="diff")
        self.assertEqual(backend._S, 1)
        self.assertEqual(backend.contrast_name, "diff")
        self.assertTrue(np.allclose(contrast, [[0.5, -0.5]]))

        with self.assertRaises(ValueError):
            backend._normalise_contrast([[0.0, 0.0]], expected_width=2)

    def test_approximate_facade_loads_params_and_creates_default_contrast(self):
        data = {
            "X_spatial": np.array([[1.0], [2.0], [3.0]]),
            "Z": np.array([[0.1], [0.2], [0.3], [0.4]]),
            "Y": np.ones((4, 3)),
        }
        params = {"beta": np.zeros((4, 1))}

        inference = BrainInference(
            model="SpatialBrainLesion",
            marginal_dist="Poisson",
            link_func="log",
            regression_terms=None,
            inference_type="approximate",
        )
        inference.load_params(data=data, params=params)
        inference.create_contrast()

        self.assertEqual(inference.B.shape, (3, 2))
        self.assertEqual(inference.Z.shape, (4, 2))
        self.assertEqual(inference.contrast_vector.shape, (1, 2))

    def test_design_loading_and_default_contrast_helpers(self):
        backend = self._backend()
        data = {
            "X_spatial": np.array([[1.0], [2.0], [3.0]]),
            "Z": np.array([[0.1], [0.2], [0.3], [0.4]]),
            "Y": np.ones((4, 3)),
        }

        backend._load_flat_design(data)

        self.assertEqual(backend.B.shape, (3, 2))
        self.assertEqual(backend.Z.shape, (4, 2))
        self.assertTrue(np.allclose(backend.B[:, -1], 1.0))
        self.assertTrue(np.allclose(backend._default_group_contrast(3), [[1, -1, 0], [0, 1, -1]]))
        self.assertTrue(np.allclose(backend._default_covariate_contrast(3, index=1), [[0, 1, 0]]))
        self.assertTrue(np.allclose(backend._ukb_age_contrast(5, polynomial_order=1), [[0, 1, 0, 0, 0]]))

    def test_contrast_design_and_variance_helpers(self):
        backend = self._backend()
        B = np.array([[1.0, 0.5], [0.2, 1.0], [1.5, -0.1]])
        contrast = np.array([[1.0, 0.0], [0.0, 1.0]])
        contrast_design = backend.contrast_design(contrast, B)
        covariance = np.eye(4)

        variance = backend.full_covariance_contrast_variance(contrast_design, covariance)
        block_variance = backend.blockwise_contrast_variance(
            B, np.array([[1.0, 0.0]]), [np.eye(2), 2.0 * np.eye(2)]
        )

        self.assertEqual(contrast_design.shape, (2, 3, 4))
        self.assertEqual(variance.shape, (2, 3))
        self.assertTrue(np.all(variance >= 0))
        self.assertEqual(block_variance.shape, (1, 3))

    def test_fisher_and_sandwich_covariance_helpers(self):
        backend = self._backend(link_func="log")
        Z = np.array([[1.0, 0.1], [1.0, 0.2], [1.0, 0.3]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        beta = np.zeros((2, 2))
        Y = np.ones((3, 2))
        mu = np.ones((3, 2))

        cov_beta, fitted_mu = backend.mass_univariate_fisher_covariance(Z, beta)
        fisher = backend.compute_fisher_information(Z, B, mu.reshape(-1, 1), use_dask=False)
        fisher_cov = backend.fisher_covariance(fisher)
        sandwich_cov = backend.sandwich_covariance(Z, B, Y, mu, ridge=1e-6)

        self.assertEqual(cov_beta.shape, (2, 2, 2))
        self.assertEqual(fitted_mu.shape, (3, 2))
        self.assertEqual(fisher.shape, (4, 4))
        self.assertEqual(fisher_cov.shape, (4, 4))
        self.assertEqual(sandwich_cov.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(sandwich_cov)))


class TestNumericalUtilities(unittest.TestCase):
    """Tests for important linear-algebra utilities."""

    def setUp(self):
        rng = np.random.default_rng(2024)
        self.Z = rng.normal(size=(4, 3))
        self.B = rng.normal(size=(5, 2))
        self.beta = rng.normal(size=(6, 1)) * 0.1
        self.mu = np.exp(kronecker_vector_product(self.Z, self.B, self.beta))

    def test_kronecker_vector_product_matches_explicit_kron(self):
        expected = np.kron(self.Z, self.B) @ self.beta
        actual = kronecker_vector_product(self.Z, self.B, self.beta)
        self.assertTrue(np.allclose(actual, expected))

    def test_efficient_kronT_diag_kron_matches_explicit_design(self):
        design = np.kron(self.Z, self.B)
        expected = design.T @ (self.mu * design)
        actual = efficient_kronT_diag_kron(self.Z, self.B, self.mu)
        self.assertTrue(np.allclose(actual, expected))

    def test_robust_inverse_helpers_return_expected_shapes(self):
        fisher = efficient_kronT_diag_kron(self.Z, self.B, self.mu)
        fisher += 1e-3 * np.eye(fisher.shape[0])

        inverse = robust_inverse(fisher)
        contrast_variance = robust_inverse_generalised(fisher, np.eye(fisher.shape[0])[:2])

        self.assertEqual(inverse.shape, fisher.shape)
        self.assertEqual(contrast_variance.shape, (2, 1))
        self.assertTrue(np.all(np.isfinite(inverse)))
        self.assertTrue(np.all(np.isfinite(contrast_variance)))

    def test_compute_mu_and_mu_summary_exact_modes(self):
        mu = compute_mu(self.Z, self.B, self.beta, mode="exact")
        mu_mean, mu_std = SpatialGLM_compute_mu_mean(self.Z, self.B, self.beta, mode="exact")
        p_mean = SpatialGLM_compute_P_mean(self.Z, self.B, self.beta, mode="exact")

        reshaped = mu.reshape(self.Z.shape[0], self.B.shape[0])
        self.assertTrue(np.allclose(mu, self.mu))
        self.assertTrue(np.allclose(mu_mean, reshaped.mean(axis=0)))
        self.assertTrue(np.allclose(mu_std, reshaped.std(axis=0)))
        self.assertTrue(np.allclose(p_mean, (reshaped * np.exp(-reshaped)).mean(axis=0)))

    def test_compute_gradient_exact_matches_explicit_formula(self):
        Y = np.ones((self.Z.shape[0], self.B.shape[0]))
        gradient = compute_gradient(self.Z, self.B, self.beta, Y, mode="exact")
        design = np.kron(self.Z, self.B)
        expected = -(design.T @ Y.reshape(-1, 1) - design.T @ self.mu)

        self.assertTrue(np.allclose(gradient, expected))

    def test_compute_preconditioner_exact_and_approximate_shapes(self):
        exact = compute_preconditioner(self.Z, self.B, beta=self.beta, mode="exact")
        mu_z = np.ones((self.Z.shape[0], 1))
        mu_x = np.ones((self.B.shape[0], 1))
        approximate = compute_preconditioner(
            self.Z, self.B, mu_Z=mu_z, mu_X=mu_x, mode="approximate"
        )

        self.assertEqual(exact.shape, (self.beta.size, self.beta.size))
        self.assertEqual(approximate.shape, (self.beta.size, self.beta.size))
        self.assertTrue(np.all(np.isfinite(exact)))
        self.assertTrue(np.all(np.isfinite(approximate)))

    def test_log_poisson_likelihood_dense_sparse_and_dask(self):
        Y = np.ones_like(self.mu)
        dense_value = log_poisson_likelihood(self.mu, Y)
        sparse_value = log_poisson_likelihood(self.mu, sparse.csr_matrix(Y))
        dask_value = log_poisson_likelihood(self.mu, Y, use_dask=True).compute()

        self.assertTrue(np.isclose(dense_value, sparse_value))
        self.assertTrue(np.isclose(dense_value, dask_value))


if __name__ == "__main__":
    unittest.main(verbosity=2)