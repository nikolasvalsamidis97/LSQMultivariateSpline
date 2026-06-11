Least-Squares Multivariate Splines
==================================

``LSQMultivariateSpline`` fits a tensor-product B-spline to scattered or
gridded observations by solving a weighted least-squares problem.

For input coordinates ``x`` and observations ``y``, the fitted spline has the
form

.. math::

   S(x_0, \ldots, x_{d-1}) =
   \sum_{i_0} \cdots \sum_{i_{d-1}}
   c_{i_0, \ldots, i_{d-1}}
   B_{i_0}(x_0) \cdots B_{i_{d-1}}(x_{d-1}).

The tensor-product basis functions are built independently along each input
axis, then multiplied together to form the design matrix.

Basic Usage
-----------

For one-dimensional data, pass a one-dimensional ``x`` array and a matching
``y`` array:

.. code-block:: python

   import numpy as np
   from lsq_multivariate_spline import LSQMultivariateSpline

   x = np.linspace(0.0, 1.0, 100)
   y = np.sin(6.0 * x)

   spline = LSQMultivariateSpline(x, y, t=8, k=3)
   values = spline(np.linspace(0.1, 0.9, 20))

Here ``t=8`` requests eight polynomial pieces. Alternatively, ``t`` may be an
explicit array of interior knot locations.

Two-Dimensional Data
--------------------

For scattered two-dimensional data, pass ``x`` with shape
``(n_samples, 2)``:

.. code-block:: python

   rng = np.random.default_rng(0)
   x0 = rng.uniform(-1.0, 1.0, 300)
   x1 = rng.uniform(-1.0, 1.0, 300)
   x = np.column_stack((x0, x1))
   y = x0**2 - x1**2

   spline = LSQMultivariateSpline(x, y, t=[6, 6], k=3)
   point_values = spline([[0.25, -0.5], [0.5, 0.5]])

The same convention extends to three or more dimensions.

Gridded Data
------------

For tensor-grid data, use :meth:`LSQMultivariateSpline.from_grid`:

.. code-block:: python

   x0 = np.linspace(-1.0, 1.0, 20)
   x1 = np.linspace(-1.0, 1.0, 25)
   x0_grid, x1_grid = np.meshgrid(x0, x1, indexing="ij")
   y = np.sin(x0_grid) + np.cos(x1_grid)

   spline = LSQMultivariateSpline.from_grid((x0, x1), y, t=[6, 6], k=3)

Derivatives
-----------

Derivatives are evaluated with ``nu``:

.. code-block:: python

   d_dx0 = spline([[0.25, -0.5]], nu=[1, 0])
   d_dx1 = spline([[0.25, -0.5]], nu=[0, 1])

For one-dimensional splines, ``nu=1`` evaluates the first derivative.

Smoothing
---------

By default the class solves the ordinary weighted least-squares problem

.. math::

   \min_c \|W(Sc - y)\|^2.

With ``smoothing > 0``, a finite-difference roughness penalty is added:

.. math::

   \min_c \|W(Sc - y)\|^2 + \lambda \|Dc\|^2.

The parameter ``smoothing`` is :math:`\lambda`. Larger values produce smoother
coefficient tensors but allow larger training residuals.

Numerical Behavior
------------------

The implementation rejects several numerically unsafe cases:

* unregularized fits with fewer samples than coefficients,
* unsupported tensor-product basis functions,
* rank-deficient dense least-squares systems,
* evaluation points outside the fitted ``bbox``,
* dense design matrices above the internal safety threshold.

Use ``sparse=True`` for larger tensor-product bases.

API Reference
-------------

.. autoclass:: lsq_multivariate_spline.LSQMultivariateSpline
   :members:
   :undoc-members:
   :show-inheritance:
