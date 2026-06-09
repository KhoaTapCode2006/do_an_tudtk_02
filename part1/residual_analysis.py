"""Generate the four residual diagnostic plots.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))  
main_dir = os.path.dirname(current_dir)  

if main_dir not in sys.path:
    sys.path.append(main_dir)

from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ridge_lasso import (
    _matmul,
    _solve_linear_system,
    _transpose,
    ridge_fit
)


def _hat_diagonal(X):
    xtx = _matmul(_transpose(X), X)
    n_features = len(xtx)

    #Kiểm tra và xử lý ma trận suy biến (Ridge Jitter) ──
    is_singular = False
    for i in range(n_features):
        if abs(xtx[i][i]) < 1e-9:
            is_singular = True
            break
            
    if is_singular or n_features > len(X): 
        # Cộng thêm một lượng nhiễu nhỏ vào đường chéo chính để biến đổi thành ma trận khả nghịch
        for i in range(n_features):
            xtx[i][i] += 1e-7

    diagonal = []
    for row in X:
        try:
            solved = _solve_linear_system(xtx, row)
        except ValueError:
            # Nếu giải hệ vẫn lỗi, thêm jitter mạnh hơn 
            xtx_fixed = [r[:] for r in xtx]
            for i in range(n_features):
                xtx_fixed[i][i] += 1e-5
            solved = _solve_linear_system(xtx_fixed, row)

        diagonal.append(
            sum(value * solved_item for value, solved_item in zip(row, solved))
        )

    return np.asarray(diagonal, dtype=float)


def residual_plots(X, y, beta_hat, save_path, show=False):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    beta_hat = np.asarray(beta_hat, dtype=float).reshape(-1)

    y_hat = X @ beta_hat
    residuals = y - y_hat
    n_obs, n_params = X.shape
    sigma_hat = np.sqrt(np.sum(residuals**2) / max(n_obs - n_params, 1))
    leverage = _hat_diagonal(np.asarray(X).tolist())
    standardized = residuals / (sigma_hat * np.sqrt(np.maximum(1.0 - leverage, 1e-12)))

    order = np.argsort(standardized)
    sorted_residuals = standardized[order]
    normal = NormalDist()
    theoretical = np.array(
        [normal.inv_cdf((i - 0.5) / n_obs) for i in range(1, n_obs + 1)]
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    axes[0, 0].scatter(y_hat, residuals, alpha=0.75, edgecolor="black", linewidth=0.4)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].scatter(
        theoretical,
        sorted_residuals,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
    )
    qq_min = min(theoretical.min(), sorted_residuals.min())
    qq_max = max(theoretical.max(), sorted_residuals.max())
    axes[0, 1].plot(
        [qq_min, qq_max],
        [qq_min, qq_max],
        color="red",
        linestyle="--",
        linewidth=1,
    )
    axes[0, 1].set_title("Normal Q-Q")
    axes[0, 1].set_xlabel("Theoretical quantiles")
    axes[0, 1].set_ylabel("Standardized residuals")
    axes[0, 1].grid(alpha=0.25)

    scale_location = np.sqrt(np.abs(standardized))
    axes[1, 0].scatter(
        y_hat,
        scale_location,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1, 0].set_title("Scale-Location")
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel("sqrt(|Standardized residuals|)")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].scatter(
        leverage,
        standardized,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1, 1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1, 1].axhline(2, color="gray", linestyle=":", linewidth=1)
    axes[1, 1].axhline(-2, color="gray", linestyle=":", linewidth=1)
    axes[1, 1].set_title("Residuals vs Leverage")
    axes[1, 1].set_xlabel("Leverage")
    axes[1, 1].set_ylabel("Standardized residuals")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle("Four Residual Diagnostic Plots", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def make_demo_data(seed=42, n_samples=120):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n_samples)
    x2 = 0.65 * x1 + rng.normal(scale=0.75, size=n_samples)
    X = np.column_stack([np.ones(n_samples), x1, x2])

    noise_scale = 0.35 + 0.25 * np.abs(x1)
    noise = rng.normal(scale=noise_scale, size=n_samples)
    y = 3.0 + 2.0 * x1 - 1.4 * x2 + noise
    return X, y


if __name__ == "__main__":
    X_demo, y_demo = make_demo_data()
    beta_demo = ridge_fit(X_demo, y_demo, lam=0.0)
    save_path = os.path.join(main_dir, "img", "residual_diagnostics.png")
    residual_plots(
        X_demo,
        y_demo,
        beta_demo,
        save_path,
        show=False,
    )
    print("Saved: img/residual_diagnostics.png")
