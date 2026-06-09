
import numpy as np
import unittest
import math

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))  
main_dir = os.path.dirname(current_dir)   


if main_dir not in sys.path:
    sys.path.append(main_dir)

def _as_matrix(values, name):
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    if hasattr(values, "tolist"):
        values = values.tolist()

    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix.")

    first = values[0]
    if not isinstance(first, (list, tuple)):
        matrix = [[float(item)] for item in values]
    else:
        matrix = [[float(item) for item in row] for row in values]

    n_cols = len(matrix[0])
    if n_cols == 0:
        raise ValueError(f"{name} must have at least one column.")
    for row in matrix:
        if len(row) != n_cols:
            raise ValueError(f"All rows of {name} must have the same length.")
    return matrix


def _as_vector(values, name):
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    if hasattr(values, "tolist"):
        values = values.tolist()

    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty vector.")

    if isinstance(values[0], (list, tuple)):
        if all(len(row) == 1 for row in values):
            vector = [row[0] for row in values]
        elif len(values) == 1:
            vector = values[0]
        else:
            raise ValueError(f"{name} must be a vector or an n x 1 matrix.")
    else:
        vector = values
    return [float(item) for item in vector]


def _feature_names(values, n_cols):
    if hasattr(values, "columns") and len(values.columns) == n_cols:
        return [str(column) for column in values.columns]
    return [f"X{index + 1}" for index in range(n_cols)]


def _transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def _matmul(left, right):
    right_t = _transpose(right)
    return [
        [sum(a * b for a, b in zip(row, col)) for col in right_t]
        for row in left
    ]


def _matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def _solve_linear_system(matrix, vector, tol=1e-12):
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("The coefficient matrix must be square.")
    if len(vector) != n:
        raise ValueError("The right-hand side vector has incompatible length.")

    aug = [row[:] + [float(vector[i])] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot_row][col]) < tol:
            raise ValueError("Singular matrix.")

        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        aug[col] = [item / pivot for item in aug[col]]

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) > tol:
                aug[row] = [
                    item - factor * pivot_item
                    for item, pivot_item in zip(aug[row], aug[col])
                ]

    return [row[-1] for row in aug]


def _ols_fit(design, target):
    xt = _transpose(design)
    xtx = _matmul(xt, design)
    xty = _matvec(xt, target)
    return _solve_linear_system(xtx, xty)


def _rss(actual, predicted):
    return sum((a - p) ** 2 for a, p in zip(actual, predicted))


def _tss(values):
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values)


def _has_intercept_column(matrix, tol=1e-12):
    return all(abs(row[0] - 1.0) < tol for row in matrix)


def vif(X):
    """Return the variance inflation factor of each independent variable.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Matrix of independent variables. Do not include the intercept column.

    Returns
    -------
    dict
        Mapping from variable name to VIF value. Perfect multicollinearity is
        reported as float("inf").
    """

    names = _feature_names(X, len(X.columns) if hasattr(X, "columns") else 0)
    matrix = _as_matrix(X, "X")
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    if len(names) != n_cols:
        names = _feature_names(X, n_cols)
    if n_rows < 2:
        raise ValueError("X must contain at least two observations.")
    if n_cols == 1:
        return {names[0]: 1.0}

    result = {}
    for target_col in range(n_cols):
        target = [row[target_col] for row in matrix]
        other_cols = [
            index for index in range(n_cols) if index != target_col
        ]
        design = [
            [1.0] + [row[index] for index in other_cols]
            for row in matrix
        ]

        total = _tss(target)
        if total < 1e-12:
            result[names[target_col]] = float("inf")
            continue

        try:
            beta = _ols_fit(design, target)
            predicted = _matvec(design, beta)
            r_squared = 1.0 - _rss(target, predicted) / total
        except ValueError:
            result[names[target_col]] = float("inf")
            continue

        if r_squared >= 1.0 - 1e-10:
            result[names[target_col]] = float("inf")
        else:
            r_squared = max(0.0, min(r_squared, 1.0 - 1e-10))
            result[names[target_col]] = 1.0 / (1.0 - r_squared)

    return result


def ridge_fit(X, y, lam):
    """Estimate Ridge regression coefficients.

    The function solves

        beta = (X^T X + lam P)^(-1) X^T y,

    where P is the identity matrix except P[0, 0] = 0 when the first column of
    X is an intercept column of ones. This keeps the intercept unpenalized.
    """

    if lam < 0:
        raise ValueError("lam must be non-negative.")

    matrix = _as_matrix(X, "X")
    target = _as_vector(y, "y")
    if len(matrix) != len(target):
        raise ValueError("X and y must have the same number of rows.")

    xt = _transpose(matrix)
    xtx = _matmul(xt, matrix)
    xty = _matvec(xt, target)

    first_penalized = 1 if _has_intercept_column(matrix) else 0
    for index in range(first_penalized, len(xtx)):
        xtx[index][index] += float(lam)

    try:
        return _solve_linear_system(xtx, xty)
    except ValueError as exc:
        raise ValueError(
            "The Ridge system is singular. Try lam > 0 or remove duplicate "
            "columns from X."
        ) from exc


def ridge_trace(X, y, lambdas=None, save_path=None, show=True):
    """Plot Ridge coefficients for a sequence of lambda values."""

    if lambdas is None:
        lambdas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    lambdas = [float(value) for value in lambdas]
    coefficients = [ridge_fit(X, y, value) for value in lambdas]

    import matplotlib.pyplot as plt

    n_coefficients = len(coefficients[0])
    for index in range(n_coefficients):
        plt.plot(
            lambdas,
            [row[index] for row in coefficients],
            marker="o",
            label=f"beta_{index}",
        )

    plt.xscale("symlog", linthresh=0.01)
    plt.xlabel("lambda")
    plt.ylabel("coefficient value")
    plt.title("Ridge trace")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    return {"lambdas": lambdas, "coefficients": coefficients}


def reference_vif(X):
    X = np.asarray(X, dtype=float)
    n_rows, n_cols = X.shape
    result = {}

    for target_col in range(n_cols):
        target = X[:, target_col]
        others = np.delete(X, target_col, axis=1)
        design = np.column_stack([np.ones(n_rows), others])

        total = np.sum((target - target.mean()) ** 2)
        if total < 1e-12:
            result[f"X{target_col + 1}"] = math.inf
            continue

        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        predicted = design @ beta
        rss = np.sum((target - predicted) ** 2)
        r_squared = 1.0 - rss / total
        result[f"X{target_col + 1}"] = (
            math.inf if r_squared >= 1.0 - 1e-10 else 1.0 / (1.0 - r_squared)
        )

    return result


def reference_ridge(X, y, lam):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    penalty = np.eye(X.shape[1])
    if np.allclose(X[:, 0], 1.0):
        penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * penalty, X.T @ y)


@unittest.skipIf(np is None, "NumPy is required for random-data tests.")
class TestVIF(unittest.TestCase):
    def test_vif_matches_numpy_lstsq_on_random_data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(120, 3))

        got = vif(X)
        expected = reference_vif(X)

        for key in expected:
            self.assertAlmostEqual(got[key], expected[key], places=7)

    def test_vif_detects_perfect_multicollinearity(self):
        rng = np.random.default_rng(123)
        x1 = rng.normal(size=80)
        x2 = 3.0 * x1
        x3 = rng.normal(size=80)
        X = np.column_stack([x1, x2, x3])

        got = vif(X)

        self.assertTrue(math.isinf(got["X1"]) or got["X1"] > 1e8)
        self.assertTrue(math.isinf(got["X2"]) or got["X2"] > 1e8)


@unittest.skipIf(np is None, "NumPy is required for NumPy/sklearn verification.")
class TestRidgeFit(unittest.TestCase):
    def test_ridge_fit_matches_numpy_closed_form(self):
        rng = np.random.default_rng(7)
        X_raw = rng.normal(size=(150, 3))
        X = np.column_stack([np.ones(X_raw.shape[0]), X_raw])
        beta_true = np.array([2.0, -1.5, 0.7, 3.2])
        y = X @ beta_true + rng.normal(scale=0.2, size=X.shape[0])
        lam = 2.5

        got = np.asarray(ridge_fit(X, y, lam))
        expected = reference_ridge(X, y, lam)

        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    def test_ridge_fit_matches_sklearn_when_available(self):
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            self.skipTest("scikit-learn is not installed.")

        rng = np.random.default_rng(99)
        X_raw = rng.normal(size=(100, 2))
        beta_true = np.array([4.0, 1.2, -2.0])
        y = beta_true[0] + X_raw @ beta_true[1:] + rng.normal(
            scale=0.3, size=X_raw.shape[0]
        )
        lam = 1.75

        design = np.column_stack([np.ones(X_raw.shape[0]), X_raw])
        got = np.asarray(ridge_fit(design, y, lam))

        model = Ridge(alpha=lam, fit_intercept=True)
        model.fit(X_raw, y)
        expected = np.r_[model.intercept_, model.coef_]

        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":

    X_demo = [
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 2.0],
        [1.0, 2.0, 2.5],
        [1.0, 3.0, 4.0],
        [1.0, 4.0, 4.5],
    ]
    y_demo = [1.0, 2.8, 4.1, 5.9, 7.2]
    print("VIF:", vif([row[1:] for row in X_demo]))
    print("Ridge beta:", ridge_fit(X_demo, y_demo, lam=1.0))
    save_path = os.path.join(main_dir, "img", "ridge_trace.png")
    ridge_trace(
        X_demo,
        y_demo,
        save_path=save_path,
        lambdas=[0.0, 0.1, 1.0, 10.0],
        show=True,
    )
    unittest.main(verbosity=2)
    

