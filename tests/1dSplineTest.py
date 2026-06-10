import matplotlib.pyplot as plt
import numpy as np

from lsq_multivariate_spline import LSQMultivariateSpline


def test_lsq_multivariate_spline_1d():
    """Plot noisy 1D data before fitting a spline."""
    rng = np.random.default_rng(0)
    x0 = np.sort(rng.uniform(0, 8 * np.pi, 100))
    y_true = np.sin(x0)
    noise = 0.1 * rng.normal(size=y_true.shape)
    y_noisy = y_true + noise

    interior_knots = np.linspace(x0.min(), x0.max(), 14)[1:-1]
    spline = LSQMultivariateSpline(
        x=x0,
        y=y_noisy,
        t=interior_knots,
        w=None,
        bbox=None,
        k=3,
        eps=1e-6,
    )

    x_fit = np.linspace(x0.min(), x0.max(), 600)
    y_fit = spline(x_fit)

    fig, ax = plt.subplots()
    ax.plot(x0, y_true, color="black", linewidth=1.5, label="True curve")
    ax.scatter(x0, y_noisy, s=18, label="Noisy samples")
    ax.plot(x_fit, y_fit, color="tab:red", linewidth=2.0, label="Spline fit")
    ax.set_title("Noisy 1D samples")
    ax.set_xlabel("x0")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    plt.show()

    print(f"Weighted residual: {spline.get_residual():.6g}")
    print(f"Number of coefficients: {spline.get_coeffs().shape[0]}")


if __name__ == "__main__":
    test_lsq_multivariate_spline_1d()
