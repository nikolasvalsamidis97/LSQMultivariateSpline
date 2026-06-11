import numpy as np
import pytest

from lsq_multivariate_spline import LSQMultivariateSpline


def test_exact_quadratic_1d():
    x = np.linspace(-1.0, 1.0, 25)
    y = 1.5 * x**2 - 0.25 * x + 2.0

    spline = LSQMultivariateSpline(x=x, y=y, t=1, k=2)
    x_eval = np.linspace(-0.8, 0.8, 9)

    np.testing.assert_allclose(
        spline(x_eval),
        1.5 * x_eval**2 - 0.25 * x_eval + 2.0,
        atol=1e-12,
    )


def test_explicit_full_knot_construction():
    x = np.linspace(0.0, 1.0, 20)
    y = x.copy()
    spline = LSQMultivariateSpline(x=x, y=y, t=[0.25, 0.5, 0.75], k=2)

    np.testing.assert_allclose(
        spline.get_knots()[0],
        [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0],
    )


def test_scalar_eval_returns_scalar_for_1d():
    x = np.linspace(0.0, 1.0, 20)
    y = 2.0 * x + 1.0
    spline = LSQMultivariateSpline(x=x, y=y, t=1, k=1)

    value = spline(0.25)

    assert np.ndim(value) == 0
    np.testing.assert_allclose(value, 1.5)


def test_coordinate_list_input_form():
    x0 = np.array([-1.0, -1.0, 1.0, 1.0])
    x1 = np.array([-1.0, 1.0, -1.0, 1.0])
    x = np.vstack((x0, x1))
    y = 1.0 + x0 - 2.0 * x1

    spline = LSQMultivariateSpline(x=x, y=y, t=[1, 1], k=1)

    np.testing.assert_allclose(spline([[0.5, -0.5]]), [2.5], atol=1e-12)


def test_exact_plane_2d_and_derivatives():
    x0_axis = np.linspace(-1.0, 1.0, 7)
    x1_axis = np.linspace(-1.5, 1.5, 8)
    x0_mesh, x1_mesh = np.meshgrid(x0_axis, x1_axis, indexing="ij")
    x = np.column_stack((x0_mesh.ravel(), x1_mesh.ravel()))
    y = 2.0 * x[:, 0] - 3.0 * x[:, 1] + 1.0

    spline = LSQMultivariateSpline(x=x, y=y, t=[1, 1], k=1)
    x_eval = np.array(
        [
            [-0.5, -0.25],
            [0.0, 0.0],
            [0.75, 1.0],
        ]
    )

    expected = 2.0 * x_eval[:, 0] - 3.0 * x_eval[:, 1] + 1.0
    np.testing.assert_allclose(spline(x_eval), expected, atol=1e-12)
    np.testing.assert_allclose(spline(x_eval, nu=[1, 0]), 2.0, atol=1e-12)
    np.testing.assert_allclose(spline(x_eval, nu=[0, 1]), -3.0, atol=1e-12)


def test_2d_derivative_matches_finite_difference():
    x0 = np.linspace(-1.0, 1.0, 12)
    x1 = np.linspace(-1.0, 1.0, 13)
    x0_mesh, x1_mesh = np.meshgrid(x0, x1, indexing="ij")
    y = np.sin(x0_mesh) + np.cos(x1_mesh)
    spline = LSQMultivariateSpline.from_grid((x0, x1), y, t=[4, 4], k=3)

    point = np.array([[0.2, -0.3]])
    step = 1e-6
    forward = spline(point + [[step, 0.0]])
    backward = spline(point - [[step, 0.0]])
    finite_difference = (forward - backward) / (2.0 * step)

    np.testing.assert_allclose(
        spline(point, nu=[1, 0]),
        finite_difference,
        rtol=1e-5,
        atol=1e-7,
    )


def test_derivative_order_above_degree_is_zero():
    x = np.linspace(0.0, 1.0, 20)
    y = 2.0 * x + 1.0
    spline = LSQMultivariateSpline(x=x, y=y, t=1, k=1)

    np.testing.assert_allclose(spline([0.25, 0.5], nu=2), 0.0)


def test_exact_linear_time_3d():
    x0_axis = np.linspace(-1.0, 1.0, 5)
    x1_axis = np.linspace(-1.0, 1.0, 6)
    time_axis = np.linspace(0.0, 4.0, 5)
    x0_mesh, x1_mesh, time_mesh = np.meshgrid(
        x0_axis,
        x1_axis,
        time_axis,
        indexing="ij",
    )
    y = 1.0 + x0_mesh + 2.0 * x1_mesh - 0.5 * time_mesh

    spline = LSQMultivariateSpline.from_grid(
        (x0_axis, x1_axis, time_axis),
        y,
        t=[1, 1, 1],
        k=1,
    )
    x_eval = np.array(
        [
            [0.2, -0.4, 1.5],
            [-0.8, 0.5, 3.0],
        ]
    )
    expected = 1.0 + x_eval[:, 0] + 2.0 * x_eval[:, 1] - 0.5 * x_eval[:, 2]

    np.testing.assert_allclose(spline(x_eval), expected, atol=1e-12)


def test_weights_are_applied_to_residual():
    x = np.linspace(0.0, 1.0, 8)
    y = np.array([1.0, 1.2, 1.6, 2.1, 2.4, 2.8, 3.3, 3.7])
    w = np.linspace(1.0, 2.0, x.size)

    spline = LSQMultivariateSpline(x=x, y=y, t=1, k=1, w=w)
    residual = np.sum((w * (spline(x) - y)) ** 2)

    np.testing.assert_allclose(spline.get_residual(), residual)


def test_explicit_bbox_flat_form():
    x = np.linspace(0.2, 0.8, 20)
    y = x.copy()

    spline = LSQMultivariateSpline(x=x, y=y, t=[0.5], bbox=[0.0, 1.0], k=1)

    np.testing.assert_allclose(spline.get_knots()[0], [0.0, 0.0, 0.5, 1.0, 1.0])


def test_automatic_knot_count_for_uniform_data():
    x = np.linspace(0.0, 10.0, 101)
    y = np.sin(x)

    spline = LSQMultivariateSpline(x=x, y=y, t=4, k=3)
    full_knots = spline.get_knots()[0]

    np.testing.assert_allclose(full_knots[4:-4], [2.5, 5.0, 7.5])


def test_matches_scipy_lsq_univariate_spline():
    interpolate = pytest.importorskip("scipy.interpolate")

    x = np.linspace(0.0, 2.0 * np.pi, 60)
    y = np.sin(x) + 0.25 * np.cos(2.0 * x)
    interior_knots = np.linspace(x.min(), x.max(), 7)[1:-1]

    ours = LSQMultivariateSpline(x=x, y=y, t=interior_knots, k=3)
    scipy_spline = interpolate.LSQUnivariateSpline(
        x,
        y,
        interior_knots,
        k=3,
    )
    x_eval = np.linspace(x.min(), x.max(), 25)

    np.testing.assert_allclose(ours(x_eval), scipy_spline(x_eval), atol=1e-10)


def test_matches_scipy_lsq_bivariate_spline():
    interpolate = pytest.importorskip("scipy.interpolate")

    x0_axis = np.linspace(-1.0, 1.0, 10)
    x1_axis = np.linspace(-1.0, 1.0, 11)
    x0_mesh, x1_mesh = np.meshgrid(x0_axis, x1_axis, indexing="ij")
    z = np.sin(x0_mesh) + 0.25 * x1_mesh**2
    x = np.column_stack((x0_mesh.ravel(), x1_mesh.ravel()))
    tx = np.array([-0.25, 0.25])
    ty = np.array([-0.5, 0.5])

    ours = LSQMultivariateSpline(x=x, y=z.ravel(), t=[tx, ty], k=3)
    scipy_spline = interpolate.LSQBivariateSpline(
        x[:, 0],
        x[:, 1],
        z.ravel(),
        tx,
        ty,
        kx=3,
        ky=3,
    )
    x_eval = np.array([[-0.6, -0.4], [0.0, 0.0], [0.7, 0.5]])

    np.testing.assert_allclose(
        ours(x_eval),
        scipy_spline.ev(x_eval[:, 0], x_eval[:, 1]),
        atol=1e-10,
    )


def test_too_few_points_for_unregularized_fit():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="not enough data points"):
        LSQMultivariateSpline(x=x, y=y, t=[0.5], k=1)


def test_rank_deficient_design_is_rejected():
    x0 = np.linspace(0.0, 1.0, 5)
    x1 = x0.copy()
    x = np.column_stack((x0, x1))
    y = np.zeros(x0.shape)

    with pytest.raises(ValueError, match="rank deficient"):
        LSQMultivariateSpline(
            x=x,
            y=y,
            t=[1, 1],
            bbox=[[0.0, 1.0], [0.0, 1.0]],
            k=1,
        )


def test_unsupported_basis_is_rejected():
    x = np.linspace(0.0, 0.4, 6)
    y = x.copy()

    with pytest.raises(ValueError, match="support every tensor-product basis"):
        LSQMultivariateSpline(x=x, y=y, t=[0.5], bbox=[0.0, 1.0], k=1)


def test_high_dimensional_unregularized_fit_is_rejected():
    x = np.linspace(0.0, 1.0, 12)[:, None]
    x = np.repeat(x, 4, axis=1)
    y = np.zeros(x.shape[0])

    with pytest.raises(ValueError, match="not enough data points"):
        LSQMultivariateSpline(x=x, y=y, t=[4, 4, 4, 4], k=3)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k": 0}, "spline degrees must be positive"),
        ({"t": 0}, "automatic knot counts"),
        ({"w": np.ones(4)}, "same length as y"),
        ({"bbox": [1.0, 0.0]}, "lower bound"),
        ({"smoothing": -1.0}, "nonnegative"),
        ({"penalty_order": 0}, "positive integer"),
        ({"sparse": "yes"}, "sparse"),
    ],
)
def test_invalid_constructor_inputs(kwargs, message):
    x = np.linspace(0.0, 1.0, 5)
    y = x.copy()
    params = {"x": x, "y": y, "t": 1, "k": 1}
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        LSQMultivariateSpline(**params)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"y": np.ones((5, 1, 1))}, "1D or 2D"),
        ({"t": [0.25, 0.25]}, "strictly increasing"),
        ({"t": [-0.1]}, "strictly inside bbox"),
        ({"bbox": [0.2, 0.8]}, "inside bbox"),
    ],
)
def test_invalid_shape_and_domain_inputs(kwargs, message):
    x = np.linspace(0.0, 1.0, 5)
    y = x.copy()
    params = {"x": x, "y": y, "t": [0.5], "k": 1}
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        LSQMultivariateSpline(**params)


def test_check_finite_rejects_nan_values():
    x = np.linspace(0.0, 1.0, 5)
    y = x.copy()
    y[2] = np.nan

    with pytest.raises(ValueError, match="finite"):
        LSQMultivariateSpline(x=x, y=y, t=1, k=1, check_finite=True)


def test_invalid_eval_shape_and_derivative_order():
    x0 = np.linspace(0.0, 1.0, 4)
    x1 = np.linspace(0.0, 1.0, 5)
    x0_mesh, x1_mesh = np.meshgrid(x0, x1, indexing="ij")
    y = x0_mesh + x1_mesh
    spline = LSQMultivariateSpline.from_grid((x0, x1), y, t=[1, 1], k=1)

    with pytest.raises(ValueError, match="shape"):
        spline([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="nonnegative"):
        spline([[0.1, 0.2]], nu=[-1, 0])

    with pytest.raises(ValueError, match="scalar evaluation"):
        spline(0.1)


def test_evaluation_outside_bbox_is_rejected():
    x = np.linspace(0.0, 1.0, 10)
    y = x.copy()
    spline = LSQMultivariateSpline(x=x, y=y, t=1, k=1)

    with pytest.raises(ValueError, match="inside bbox"):
        spline([1.1])
