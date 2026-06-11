"""Benchmarks for LSQMultivariateSpline.

These benchmarks follow the lightweight ASV style used by scientific Python
projects: classes prepare deterministic inputs in ``setup`` and expose
``time_*`` methods for ASV to time.
"""

from __future__ import annotations

import numpy as np

from lsq_multivariate_spline import LSQMultivariateSpline
from lsq_multivariate_spline._basis import _construct_full_knots, _design_matrix


class Fit1D:
    def setup(self):
        rng = np.random.default_rng(0)
        self.x = np.sort(rng.uniform(0.0, 1.0, 500))
        self.y = np.sin(8.0 * self.x) + 0.05 * rng.normal(size=self.x.shape)
        self.t = 12

    def time_fit_1d(self):
        LSQMultivariateSpline(self.x, self.y, t=self.t, bbox=[0.0, 1.0], k=3)


class Evaluate1D:
    def setup(self):
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0.0, 1.0, 500))
        y = np.sin(8.0 * x) + 0.05 * rng.normal(size=x.shape)
        self.spline = LSQMultivariateSpline(x, y, t=12, bbox=[0.0, 1.0], k=3)
        self.x_eval = np.linspace(0.0, 1.0, 2000)

    def time_evaluate_1d(self):
        self.spline(self.x_eval)


class Fit2D:
    def setup(self):
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-1.0, 1.0, 900)
        x1 = rng.uniform(-1.0, 1.0, 900)
        self.x = np.column_stack((x0, x1))
        self.y = np.sin(x0) + np.cos(x1)
        self.t = [8, 8]

    def time_fit_2d(self):
        LSQMultivariateSpline(
            self.x,
            self.y,
            t=self.t,
            bbox=[[-1.0, 1.0], [-1.0, 1.0]],
            k=3,
        )


class Evaluate2D:
    def setup(self):
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-1.0, 1.0, 900)
        x1 = rng.uniform(-1.0, 1.0, 900)
        x = np.column_stack((x0, x1))
        y = np.sin(x0) + np.cos(x1)
        self.spline = LSQMultivariateSpline(
            x,
            y,
            t=[8, 8],
            bbox=[[-1.0, 1.0], [-1.0, 1.0]],
            k=3,
        )
        grid = np.linspace(-1.0, 1.0, 75)
        x0_eval, x1_eval = np.meshgrid(grid, grid, indexing="ij")
        self.x_eval = np.column_stack((x0_eval.ravel(), x1_eval.ravel()))

    def time_evaluate_2d(self):
        self.spline(self.x_eval)


class Fit3D:
    def setup(self):
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-1.0, 1.0, 2000)
        x1 = rng.uniform(-1.0, 1.0, 2000)
        x2 = rng.uniform(0.0, 1.0, 2000)
        self.x = np.column_stack((x0, x1, x2))
        self.y = np.sin(x0) + np.cos(x1) + x2
        self.t = [5, 5, 5]

    def time_fit_3d(self):
        LSQMultivariateSpline(
            self.x,
            self.y,
            t=self.t,
            bbox=[[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]],
            k=3,
            sparse=True,
        )


class Evaluate3D:
    def setup(self):
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-1.0, 1.0, 2000)
        x1 = rng.uniform(-1.0, 1.0, 2000)
        x2 = rng.uniform(0.0, 1.0, 2000)
        x = np.column_stack((x0, x1, x2))
        y = np.sin(x0) + np.cos(x1) + x2
        self.spline = LSQMultivariateSpline(
            x,
            y,
            t=[5, 5, 5],
            bbox=[[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]],
            k=3,
            sparse=True,
        )
        self.x_eval = rng.uniform(
            [-1.0, -1.0, 0.0],
            [1.0, 1.0, 1.0],
            (5000, 3),
        )

    def time_evaluate_3d(self):
        self.spline(self.x_eval)


class DenseSparseFit2D:
    params = [False, True]
    param_names = ["sparse"]

    def setup(self, sparse):
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-1.0, 1.0, 1200)
        x1 = rng.uniform(-1.0, 1.0, 1200)
        self.x = np.column_stack((x0, x1))
        self.y = 0.5 * (x0**2 - x1**2)
        self.sparse = sparse

    def time_fit_2d_dense_or_sparse(self, sparse):
        LSQMultivariateSpline(
            self.x,
            self.y,
            t=[8, 8],
            bbox=[[-1.0, 1.0], [-1.0, 1.0]],
            k=3,
            sparse=sparse,
        )


class DesignMatrix2D:
    params = [False, True]
    param_names = ["sparse"]

    def setup(self, sparse):
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-1.0, 1.0, 2000)
        x1 = rng.uniform(-1.0, 1.0, 2000)
        self.x = np.column_stack((x0, x1))
        bbox = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        interior = [np.linspace(-1.0, 1.0, 10)[1:-1]] * 2
        self.knots = _construct_full_knots(interior, bbox, (3, 3))
        self.degrees = (3, 3)

    def time_design_matrix_2d(self, sparse):
        _design_matrix(self.x, self.knots, self.degrees, sparse=sparse)


class ScalingWithSamplesAndKnots:
    params = ([200, 1000], [5, 10])
    param_names = ["n_samples", "n_pieces"]

    def setup(self, n_samples, n_pieces):
        rng = np.random.default_rng(0)
        self.x = np.sort(rng.uniform(0.0, 1.0, n_samples))
        self.y = np.sin(6.0 * self.x)
        self.n_pieces = n_pieces

    def time_fit_1d_scaling(self, n_samples, n_pieces):
        LSQMultivariateSpline(
            self.x,
            self.y,
            t=n_pieces,
            bbox=[0.0, 1.0],
            k=3,
        )
