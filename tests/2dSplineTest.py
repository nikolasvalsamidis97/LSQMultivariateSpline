from lsq_multivariate_spline import LSQMultivariateSpline


def test_lsq_multivariate_spline_2d():
    """Test the LSQMultivariateSpline class with a simple example."""
    # Example data: 2D noisy samples from a known function
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)
    x0 = np.random.uniform(0, 1, 100)
    x1 = np.random.uniform(0, 1, 100)
    y_true = np.sin(2 * np.pi * x0) * np.cos(2 * np.pi * x1)
    noise = 0.1 * np.random.normal(size=y_true.shape)
    y_noisy = y_true + noise

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_trisurf(
        x0,
        x1,
        y_noisy,
        cmap="viridis",
        edgecolor="none",
        alpha=0.8,
    )
    ax.scatter(x0, x1, y_noisy, color="black", s=12)
    ax.set_title("Noisy 2D samples")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("y_noisy")
    fig.colorbar(surf, ax=ax, shrink=0.7, label="y_noisy")
    fig.tight_layout()
    plt.show()

    # Fit the spline (this will raise NotImplementedError until implemented)
    x = np.column_stack((x0, x1))
    try:
        spline = LSQMultivariateSpline(
            x=x,
            y=y_noisy,
            t=[[], []],
            w=None,
            bbox=None,
            k=[3, 3],
            eps=1e-6,
        )
        print("Spline fitted successfully (this should not happen yet).")
    except NotImplementedError as e:
        print(f"Expected error: {e}")


if __name__ == "__main__":
    test_lsq_multivariate_spline_2d()
