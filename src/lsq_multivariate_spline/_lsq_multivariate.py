"""Least-squares multivariate spline approximation.

This module is the starting point for a SciPy-style implementation. The public
class mirrors the shape of ``scipy.interpolate.LSQBivariateSpline``, but the
input dimension is generalized from exactly two coordinates to ``n`` input
coordinates.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from operator import index

import numpy as np


__all__ = ["LSQMultivariateSpline"]


class LSQMultivariateSpline:
    """Weighted least-squares multivariate spline approximation.

    Parameters
    ----------
    x : array_like, shape (n_samples, n_dimensions)
        Coordinates of the input data. Each row is one sample point; each
        column is one input dimension.
    y : array_like, shape (n_samples,)
        Observed values at the sample coordinates.
    t : sequence of array_like
        Interior knot coordinates. There should be one knot vector for each
        input dimension.
    w : array_like, shape (n_samples,), optional
        Positive weights for weighted least-squares fitting.
    bbox : array_like, shape (n_dimensions, 2), optional
        Boundary of the approximation domain for each input dimension.
    k : int or sequence of int, optional
        Spline degree for each input dimension. A scalar applies the same
        degree to every dimension. Default is cubic, ``k=3``.
    eps : float, optional
        Rank threshold used by the least-squares solver.
    check_finite : bool, optional
        Whether to check input arrays for NaN or infinite values.

    Notes
    -----
    The planned fitting model is a tensor-product B-spline basis. In two
    dimensions this is analogous to bivariate spline basis terms
    ``B_i(x0) * B_j(x1)``. In ``n`` dimensions the basis terms become products
    over all input axes.
    """

    def __init__(
        self,
        x,
        y,
        t,
        w=None,
        bbox=None,
        k=3,
        eps=None,
        check_finite=False,
    ):
        self._input = _validate_input(
            x=x,
            y=y,
            t=t,
            w=w,
            bbox=bbox,
            k=k,
            eps=eps,
            check_finite=check_finite,
        )

        self._basis = None
        self._coeffs = None
        self._residual = None

        self._fit()

    def __call__(self, x, nu=None):
        """Evaluate the spline or its derivatives at given positions."""
        return self.ev(x, nu=nu)

    def ev(self, x, nu=None):
        """Evaluate the spline at scattered points.

        Parameters
        ----------
        x : array_like, shape (n_eval, n_dimensions)
            Points where the spline should be evaluated.
        nu : int or sequence of int, optional
            Derivative order for each input dimension.
        """
        if nu is not None:
            raise NotImplementedError("Derivative evaluation is not implemented yet.")

        x, was_scalar = _validate_eval_input(x, ndim=self._input.x.shape[1])

        s_matrix = _design_matrix(x, self._basis, self._input.k)
        values = s_matrix @ self._coeffs

        if was_scalar:
            return values[0]

        return values

    def get_knots(self):
        """Return the knot vectors for each input dimension."""
        return self._basis

    def get_coeffs(self):
        """Return the fitted spline coefficients."""
        return self._coeffs.copy()

    def get_residual(self):
        """Return the weighted sum of squared residuals."""
        return self._residual

    def _fit(self):
        """Fit the spline coefficients from the constructor inputs."""
        full_knots = _construct_full_knots(
            self._input.t,
            self._input.bbox,
            self._input.k,
        )
        s_matrix = _design_matrix(self._input.x, full_knots, self._input.k)
        coeffs, residual = _solve_lsq(
            s_matrix,
            self._input.y,
            w=self._input.w,
            eps=self._input.eps,
        )

        self._basis = full_knots
        self._coeffs = coeffs
        self._residual = residual


@dataclass
class _SplineInput:
    """Validated and normalized constructor inputs."""

    x: object
    y: object
    t: object
    w: object
    bbox: object
    k: object
    eps: object
    check_finite: bool


@dataclass
class _SplineState:
    """Validated state produced by fitting."""

    knots: object
    degrees: object
    coefficients: object
    residual: object


class _TensorProductBasis:
    """Tensor-product B-spline basis used by the multivariate spline."""

    def __init__(self, knots, degrees):
        self.knots = knots
        self.degrees = degrees

    def design_matrix(self, x, nu=None):
        """Build the design matrix for fitting or evaluation."""
        raise NotImplementedError("Basis matrix construction is not implemented yet.")


def _validate_input(x, y, t, w, bbox, k, eps, check_finite):
    """Validate and normalize constructor inputs."""
    y = np.asarray(y, dtype=float)

    if check_finite and not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    n_samples = y.shape[0]
    if n_samples == 0:
        raise ValueError("x and y must contain at least one sample.")

    # Convert x to matrix form: (n_samples, n_dimensions).
    x = np.asarray(x, dtype=float)

    if check_finite and not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")

    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim == 2:
        if x.shape[0] == n_samples:
            # Already in matrix form: (n_samples, n_dimensions).
            pass
        elif x.shape[1] == n_samples:
            # Coordinate-list form: (n_dimensions, n_samples).
            x = x.T
        else:
            raise ValueError(
                "x must have shape (n_samples, n_dimensions) or "
                "(n_dimensions, n_samples)."
            )
    else:
        raise ValueError("x must be a 1D or 2D array.")

    if y.shape[0] != x.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")

    n_samples, ndim = x.shape
    if ndim == 0:
        raise ValueError("x must contain at least one input dimension.")

    # Ensure w is a 1D array of positive weights. SciPy spline weights multiply
    # residuals before squaring.
    if w is None:
        w = np.ones(n_samples, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

        if check_finite and not np.all(np.isfinite(w)):
            raise ValueError("w must contain only finite values.")

        if w.ndim != 1:
            raise ValueError("w must be a 1D array.")

        if w.shape[0] != n_samples:
            raise ValueError("w must have the same length as y.")

        if np.any(w <= 0):
            raise ValueError("w must contain only positive values.")

    # Normalize spline degree k to one integer degree per dimension.
    if np.isscalar(k):
        try:
            degree = index(k)
        except TypeError as exc:
            raise ValueError("k must be an integer or a sequence of integers.") from exc

        degrees = (degree,) * ndim
    else:
        try:
            degrees = tuple(index(ki) for ki in k)
        except TypeError as exc:
            raise ValueError("k must be an integer or a sequence of integers.") from exc

    if len(degrees) != ndim:
        raise ValueError("k must be a scalar or have one value per dimension.")

    if any(ki < 1 for ki in degrees):
        raise ValueError("spline degrees must be positive integers.")

    # Normalize bbox to shape (n_dimensions, 2). If omitted, infer it from x.
    if bbox is None:
        bbox = np.column_stack((np.min(x, axis=0), np.max(x, axis=0)))
    else:
        bbox = np.asarray(bbox, dtype=float)

        if bbox.shape == (2,) and ndim == 1:
            bbox = bbox[None, :]
        elif bbox.shape == (2 * ndim,):
            bbox = bbox.reshape(ndim, 2)

        if bbox.shape != (ndim, 2):
            raise ValueError("bbox must have shape (n_dimensions, 2).")

    if check_finite and not np.all(np.isfinite(bbox)):
        raise ValueError("bbox must contain only finite values.")

    if np.any(bbox[:, 0] >= bbox[:, 1]):
        raise ValueError("each bbox lower bound must be less than its upper bound.")

    if np.any(x < bbox[:, 0]) or np.any(x > bbox[:, 1]):
        raise ValueError("all x values must lie inside bbox.")

    # Normalize interior knots to one strictly increasing vector per dimension.
    if ndim == 1:
        if _is_sequence_of_sequences(t):
            if len(t) != 1:
                raise ValueError("t must contain one knot vector per dimension.")
            knot_vectors = (np.asarray(t[0], dtype=float),)
        else:
            knot_vectors = (np.asarray(t, dtype=float),)
    else:
        if not _is_sequence_of_sequences(t) or len(t) != ndim:
            raise ValueError("t must contain one knot vector per dimension.")
        knot_vectors = tuple(np.asarray(ti, dtype=float) for ti in t)

    for axis, knots in enumerate(knot_vectors):
        if knots.ndim != 1:
            raise ValueError("each knot vector in t must be 1D.")

        if check_finite and not np.all(np.isfinite(knots)):
            raise ValueError("t must contain only finite knot values.")

        if knots.size and np.any(np.diff(knots) <= 0):
            raise ValueError("each knot vector in t must be strictly increasing.")

        lower, upper = bbox[axis]
        if knots.size and (np.any(knots <= lower) or np.any(knots >= upper)):
            raise ValueError("interior knots must lie strictly inside bbox.")

    # Validate eps for the later least-squares rank decision.
    if eps is not None:
        if not np.isscalar(eps):
            raise ValueError("eps must be a scalar.")

        eps = float(eps)
        if not np.isfinite(eps):
            raise ValueError("eps must be finite.")

        if not 0.0 < eps < 1.0:
            raise ValueError("eps must lie in the open interval (0, 1).")

    return _SplineInput(
        x=x,
        y=y,
        t=knot_vectors,
        w=w,
        bbox=bbox,
        k=degrees,
        eps=eps,
        check_finite=check_finite,
    )


def _construct_full_knots(t, bbox, degrees):
    """Construct full knot vectors from interior knots and boundaries."""
    full_knots = []

    for axis, interior_knots in enumerate(t):
        degree = degrees[axis]
        lower, upper = bbox[axis]

        knots = np.concatenate(
            (
                np.repeat(lower, degree + 1),
                interior_knots,
                np.repeat(upper, degree + 1),
            )
        )
        full_knots.append(knots)

    return tuple(full_knots)


def _bspline_basis_axis(x, knots, degree, basis_index):
    """Evaluate one 1D B-spline basis function."""
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    last_basis_index = len(knots) - degree - 2

    if degree == 0:
        left = knots[basis_index]
        right = knots[basis_index + 1]
        values = np.where((left <= x) & (x < right), 1.0, 0.0)

        # Include the final right boundary in the last basis interval.
        if basis_index == last_basis_index:
            values = np.where(x == knots[-1], 1.0, values)

        return values

    left_denominator = knots[basis_index + degree] - knots[basis_index]
    if left_denominator == 0:
        left_part = 0.0
    else:
        left_part = (
            (x - knots[basis_index])
            / left_denominator
            * _bspline_basis_axis(x, knots, degree - 1, basis_index)
        )

    right_denominator = (
        knots[basis_index + degree + 1] - knots[basis_index + 1]
    )
    if right_denominator == 0:
        right_part = 0.0
    else:
        right_part = (
            (knots[basis_index + degree + 1] - x)
            / right_denominator
            * _bspline_basis_axis(x, knots, degree - 1, basis_index + 1)
        )

    values = left_part + right_part
    if basis_index == last_basis_index:
        values = np.where(x == knots[-1], 1.0, values)

    return values


def _design_matrix_axis(x, knots, degree):
    """Build the 1D basis matrix for one input axis.

    Returns
    -------
    matrix : ndarray, shape (n_samples, n_basis_axis)
        matrix[i, j] = B_j(x_i)
    """
    n_basis = len(knots) - degree - 1
    columns = [
        _bspline_basis_axis(x, knots, degree, basis_index)
        for basis_index in range(n_basis)
    ]
    return np.column_stack(columns)


def _design_matrix(x, knots, degrees):
    """Build the tensor-product design matrix for all dimensions.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_dimensions)
        Sample coordinates.
    knots : tuple of ndarray
        Full knot vector for each dimension.
    degrees : tuple of int
        Spline degree for each dimension.

    Returns
    -------
    matrix : ndarray, shape (n_samples, n_total_basis)
        Design matrix where each column is one tensor-product basis function.
    """
    x = np.asarray(x, dtype=float)
    n_samples, ndim = x.shape

    axis_matrices = [
        _design_matrix_axis(
            x[:, axis],
            knots[axis],
            degrees[axis],
        )
        for axis in range(ndim)
    ]

    basis_counts = [matrix.shape[1] for matrix in axis_matrices]
    basis_index_combinations = itertools.product(
        *[range(count) for count in basis_counts]
    )

    columns = []
    for basis_indices in basis_index_combinations:
        column = np.ones(n_samples, dtype=float)

        for axis, basis_index in enumerate(basis_indices):
            column *= axis_matrices[axis][:, basis_index]

        columns.append(column)

    return np.column_stack(columns)


def _is_sequence_of_sequences(value):
    """Return True when value looks like a sequence of knot vectors."""
    if isinstance(value, (str, bytes)):
        return False

    try:
        items = list(value)
    except TypeError:
        return False

    if not items:
        return False

    return all(np.ndim(item) > 0 for item in items)


def _validate_eval_input(x, ndim):
    """Validate and normalize evaluation coordinates."""
    was_scalar = np.ndim(x) == 0
    x = np.asarray(x, dtype=float)

    if x.ndim == 0:
        if ndim != 1:
            raise ValueError("scalar evaluation is only valid for 1D splines.")
        x = x[None, None]
    elif x.ndim == 1:
        if ndim == 1:
            x = x[:, None]
        elif x.shape[0] == ndim:
            x = x[None, :]
        else:
            raise ValueError("x must have shape (n_eval, n_dimensions).")
    elif x.ndim == 2:
        if x.shape[1] != ndim:
            raise ValueError("x must have shape (n_eval, n_dimensions).")
    else:
        raise ValueError("x must be a scalar, 1D array, or 2D array.")

    return x, was_scalar


def _solve_lsq(design_matrix, y, w=None, eps=None):
    """Solve the weighted least-squares system for spline coefficients."""
    if w is None:
        residual_weights = 1.0
        weighted_matrix = design_matrix
        weighted_y = y
    else:
        residual_weights = w
        weighted_matrix = design_matrix * w[:, None]
        weighted_y = y * w

    coeffs, *_ = np.linalg.lstsq(weighted_matrix, weighted_y, rcond=eps)
    residual = np.sum((residual_weights * (design_matrix @ coeffs - y)) ** 2)

    return coeffs, residual
