"""Plot the noisy sample data used for spline development checks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(__file__).resolve().parent


def load_csv(filename):
    """Load one numeric CSV file from this directory."""
    return np.genfromtxt(DATA_DIR / filename, delimiter=",", names=True)


def main():
    data_1d = load_csv("noisy_spline_1d.csv")
    data_2d = load_csv("noisy_spline_2d.csv")
    data_3d = load_csv("noisy_spline_3d.csv")
    data_4d = load_csv("noisy_spline_4d.csv")
    noise_1d = data_1d["y_noisy"] - data_1d["y_true"]
    noise_4d = data_4d["y_noisy"] - data_4d["y_true"]

    fig = plt.figure(figsize=(13, 9))

    ax = fig.add_subplot(2, 3, 1)
    ax.plot(data_1d["x0"], data_1d["y_true"], color="black", linewidth=1.5)
    ax.scatter(data_1d["x0"], data_1d["y_noisy"], s=18)
    ax.set_title("1D noisy curve")
    ax.set_xlabel("x0")
    ax.set_ylabel("y")

    ax = fig.add_subplot(2, 3, 4)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.scatter(data_1d["x0"], noise_1d, s=18)
    ax.set_title("1D noise")
    ax.set_xlabel("x0")
    ax.set_ylabel("y_noisy - y_true")

    ax = fig.add_subplot(2, 3, 2)
    image = ax.scatter(data_2d["x0"], data_2d["x1"], c=data_2d["y_noisy"], s=18)
    fig.colorbar(image, ax=ax, shrink=0.8, label="y_noisy")
    ax.set_title("2D noisy surface samples")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")

    ax = fig.add_subplot(2, 3, 5, projection="3d")
    image = ax.scatter(
        data_3d["x0"],
        data_3d["x1"],
        data_3d["x2"],
        c=data_3d["y_noisy"],
        s=10,
    )
    fig.colorbar(image, ax=ax, shrink=0.65, label="y_noisy")
    ax.set_title("3D noisy scalar field")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("x2")

    ax = fig.add_subplot(2, 3, 3)
    image = ax.scatter(data_4d["x0"], data_4d["x1"], c=data_4d["y_noisy"], s=12)
    fig.colorbar(image, ax=ax, shrink=0.8, label="y_noisy")
    ax.set_title("4D projection: x0/x1")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")

    ax = fig.add_subplot(2, 3, 6)
    image = ax.scatter(data_4d["x2"], data_4d["x3"], c=noise_4d, s=12)
    fig.colorbar(image, ax=ax, shrink=0.8, label="noise")
    ax.set_title("4D projection: x2/x3 noise")
    ax.set_xlabel("x2")
    ax.set_ylabel("x3")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
