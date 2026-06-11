import numpy as np
import pytest

from lsq_multivariate_spline import LSQMultivariateSpline


def test_automatic_knots_and_derivative_1d():
    x = np.linspace(0.0, 1.0, 30)
    y = 2.0 * x + 1.0

    spline = LSQMultivariateSpline(x=x, y=y, t=1, k=1)

    x_eval = np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(spline(x_eval), 2.0 * x_eval + 1.0)
    np.testing.assert_allclose(spline(x_eval, nu=1), 2.0)


def test_vector_valued_output():
    x = np.linspace(0.0, 1.0, 30)
    y = np.column_stack((2.0 * x + 1.0, -3.0 * x + 4.0))

    spline = LSQMultivariateSpline(x=x, y=y, t=1, k=1)
    values = spline([0.25, 0.75])

    assert values.shape == (2, 2)
    expected = np.column_stack(
        (
            2.0 * np.array([0.25, 0.75]) + 1.0,
            -3.0 * np.array([0.25, 0.75]) + 4.0,
        )
    )
    np.testing.assert_allclose(values, expected)


def test_tensor_shaped_coefficients_for_vector_output():
    x0 = np.linspace(-1.0, 1.0, 5)
    x1 = np.linspace(-1.0, 1.0, 6)
    x0_mesh, x1_mesh = np.meshgrid(x0, x1, indexing="ij")
    y0 = x0_mesh + x1_mesh
    y1 = x0_mesh - x1_mesh
    y = np.stack((y0, y1), axis=-1)

    spline = LSQMultivariateSpline.from_grid((x0, x1), y, t=[1, 1], k=1)
    coeffs = spline.get_coeffs()
    coeffs_tensor = spline.get_coeffs_tensor()

    assert coeffs.shape == (4, 2)
    assert coeffs_tensor.shape == (2, 2, 2)
    np.testing.assert_allclose(coeffs_tensor.reshape(4, 2), coeffs)


def test_get_coeffs_returns_copy():
    x = np.linspace(0.0, 1.0, 20)
    y = 2.0 * x + 1.0
    spline = LSQMultivariateSpline(x=x, y=y, t=1, k=1)

    coeffs = spline.get_coeffs()
    coeffs[...] = 0.0

    np.testing.assert_allclose(spline([0.25]), [1.5])


def test_from_grid_scalar_values():
    x0 = np.linspace(-1.0, 1.0, 6)
    x1 = np.linspace(-1.0, 1.0, 7)
    x0_mesh, x1_mesh = np.meshgrid(x0, x1, indexing="ij")
    y = 0.5 * x0_mesh - 0.25 * x1_mesh + 2.0

    spline = LSQMultivariateSpline.from_grid((x0, x1), y, t=[1, 1], k=1)
    value = spline([0.5, -0.5])

    np.testing.assert_allclose(value, 2.375)


def test_from_grid_vector_values_output_first_axis():
    x0 = np.linspace(-1.0, 1.0, 5)
    x1 = np.linspace(-1.0, 1.0, 6)
    x0_mesh, x1_mesh = np.meshgrid(x0, x1, indexing="ij")
    y0 = 1.0 + x0_mesh
    y1 = 2.0 - x1_mesh
    y = np.stack((y0, y1), axis=0)

    spline = LSQMultivariateSpline.from_grid((x0, x1), y, t=[1, 1], k=1)
    value = spline([[0.25, -0.5]])

    np.testing.assert_allclose(value, [[1.25, 2.5]], atol=1e-12)


def test_from_grid_weights_are_applied():
    x = np.linspace(0.0, 1.0, 8)
    y = np.array([1.0, 1.1, 1.5, 2.0, 2.6, 3.1, 3.6, 4.2])
    w = np.linspace(1.0, 2.0, x.size)

    spline = LSQMultivariateSpline.from_grid((x,), y, w=w, t=1, k=1)
    residual = np.sum((w * (spline(x) - y)) ** 2)

    np.testing.assert_allclose(spline.get_residual(), residual)


def test_sparse_fit_matches_dense_fit():
    pytest.importorskip("scipy")

    x = np.linspace(0.0, 1.0, 40)
    y = 2.0 * x + 1.0

    dense = LSQMultivariateSpline(x=x, y=y, t=1, k=1)
    sparse = LSQMultivariateSpline(x=x, y=y, t=1, k=1, sparse=True)
    x_eval = np.linspace(0.1, 0.9, 5)

    np.testing.assert_allclose(sparse(x_eval), dense(x_eval), atol=1e-10)


def test_sparse_fit_matches_dense_fit_2d():
    pytest.importorskip("scipy")

    x0 = np.linspace(-1.0, 1.0, 7)
    x1 = np.linspace(-1.0, 1.0, 8)
    x0_mesh, x1_mesh = np.meshgrid(x0, x1, indexing="ij")
    x = np.column_stack((x0_mesh.ravel(), x1_mesh.ravel()))
    y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1]

    dense = LSQMultivariateSpline(x=x, y=y, t=[1, 1], k=1)
    sparse = LSQMultivariateSpline(x=x, y=y, t=[1, 1], k=1, sparse=True)
    x_eval = np.array([[-0.25, 0.1], [0.5, -0.8]])

    np.testing.assert_allclose(sparse(x_eval), dense(x_eval), atol=1e-10)


def test_smoothing_reduces_coefficient_roughness():
    pytest.importorskip("scipy")

    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 80)
    y = np.sin(12.0 * x) + 0.2 * rng.normal(size=x.shape)

    rough = LSQMultivariateSpline(x=x, y=y, t=14, k=3)
    smooth = LSQMultivariateSpline(x=x, y=y, t=14, k=3, smoothing=10.0)

    roughness_plain = np.sum(np.diff(rough.get_coeffs_tensor(), n=2) ** 2)
    roughness_smooth = np.sum(np.diff(smooth.get_coeffs_tensor(), n=2) ** 2)

    assert roughness_smooth < roughness_plain
    assert smooth.get_residual() > rough.get_residual()


def test_zero_smoothing_matches_plain_least_squares():
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(4.0 * x)

    plain = LSQMultivariateSpline(x=x, y=y, t=5, k=3)
    smooth_zero = LSQMultivariateSpline(x=x, y=y, t=5, k=3, smoothing=0.0)
    x_eval = np.linspace(0.0, 1.0, 12)

    np.testing.assert_allclose(smooth_zero(x_eval), plain(x_eval))
