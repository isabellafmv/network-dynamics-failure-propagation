"""
Replaces VAR (Vector Autoregression) with mechanistic ODE discovery using
SINDy (Sparse Identification of Nonlinear Dynamics). 

Uses PC informed adjacency matrix as a causal sparsity mask so that SINDy only discovers edges already deemed causally plausible.

Usage (standalone smoke-test):
    python sindy.py

Usage (imported into an existing pipeline):
    from sindy import build_causal_mask, fit_sindy_with_mask, \
                              plot_sindy_results, validate_sindy_fit
"""

from __future__ import annotations

import argparse
import textwrap
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import solve_ivp

try:
    import pysindy as ps
except ImportError as e:
    raise ImportError(
        "PySINDy is required:  pip install pysindy"
    ) from e


# 1.  DATA ADAPTER
# Normalises input from either pipeline into (X, t, var_names)

class DataAdapter:
    """Convert pipeline data into the (X, t, var_names) triple expected by SINDy.

    df : pd.DataFrame
        A time-series DataFrame where rows are time steps and columns are
        biomarker variables. Pass ``df_train`` from deep-learning.py or
        ``df_train_ts`` from cohorstudy.py directly.
    dt : float
        Sampling interval in natural units (1.0 = daily, 7.0 = weekly, …).
        SINDy uses this to compute derivatives via finite differences.
    """

    def __init__(self, df: pd.DataFrame, dt: float = 1.0):
        self.df = df.copy().reset_index(drop=True)
        self.dt = dt
        self.var_names: list[str] = list(df.columns)

    @property
    def X(self) -> np.ndarray:
        """State matrix of shape (T, n_vars)."""
        return self.df.to_numpy(dtype=float)

    @property
    def t(self) -> np.ndarray:
        """Time vector of shape (T,)."""
        T = len(self.df)
        return np.arange(T, dtype=float) * self.dt


# 2.  CAUSAL MASK BUILDER
#     Converts the Stage 2 PC adjacency matrix into a feature-library mask.

def build_causal_mask(
    adj: np.ndarray,
    var_names: list[str],
    poly_degree: int = 1,
    include_bias: bool = True,
) -> np.ndarray:
    """binary constraint mask for PySINDy from the PC adjacency matrix.

    PySINDy's optimizer discovers an ODE system: with feature library (monomials up to poly_degree) and sparse coefficient matrix (n_feat x n_vars).

    This function returns a *column-wise* mask M of shape (n_feat, n_vars) such that M[k, i] = 0 whenever feature k involves a variable x_j that has no causal path to x_i in the PC graph.  
    PySINDy's ``ConstrainedSR3`` / ``STLSQ`` can use this to enforce structural zeros throughout optimisation.

    Parameters
    ----------
    adj : np.ndarray, shape (n_vars, n_vars)
        PC CPDAG adjacency matrix ``cg.G.graph``.
        Convention (causal-learn):
          adj[j, i] ==  1  and  adj[i, j] == -1  →  directed edge i → j
          adj[i, j] == -1  and  adj[j, i] == -1  →  undirected edge i — j
    var_names : list[str]
        Variable names in the same order as columns in the data matrix.
    poly_degree : int
        Degree of the polynomial feature library (must match the library used
        for fitting).
    include_bias : bool
        Whether the library includes a constant bias term (PySINDy default).

    Returns
    -------
    mask : np.ndarray of bool, shape (n_feat, n_vars)
        True  → coefficient is *allowed* to be nonzero.
        False → coefficient is *forced* to zero (causally implausible).
    """
    n = len(var_names)

    # Derive parent sets from PC output
    # parents[i] = set of variable indices that can causally influence x_i
    # include self-regulation (i in parents[i])
    parents: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        parents[i].add(i)  # self-regulation always allowed
        for j in range(n):
            if i == j:
                continue
            a_ij = adj[i, j]
            a_ji = adj[j, i]
            # causal-learn convention:
            #   adj[j, i] = 1  AND  adj[i, j] = -1  →  directed edge  i → j
            # So j is a PARENT of i (j → i) when:
            #   adj[i, j] = 1  AND  adj[j, i] = -1
            if a_ij == 1 and a_ji == -1:
                # j → i  (j is a directed parent of i)
                parents[i].add(j)
            elif a_ij == -1 and a_ji == -1:
                # undirected i — j  (keep both as candidate parents)
                parents[i].add(j)
            elif a_ij == 1 and a_ji == 1:
                # bidirected (rare in plain PC): allow both directions
                parents[i].add(j)

    # Enumerate polynomial library features
    # PySINDy's PolynomialLibrary generates terms in grlex order.
    # We reproduce the ordering here to assign parent sets to features.
    import itertools

    feature_names_order: list[frozenset] = []
    for degree in range(0 if include_bias else 1, poly_degree + 1):
        for combo in itertools.combinations_with_replacement(range(n), degree):
            feature_names_order.append(frozenset(combo))  # multiset as frozenset

    n_feat = len(feature_names_order)

    # Build the mask
    # A feature is allowed for variable i if ALL variables in the feature's
    # monomial are within the parent set of i (or are the bias/constant term).
    mask = np.ones((n_feat, n), dtype=bool)

    for feat_idx, combo_set in enumerate(feature_names_order):
        if len(combo_set) == 0:
            # Bias term → always allowed (models intrinsic drift)
            continue
        for var_idx in range(n):
            allowed = parents[var_idx]
            # Every variable appearing in this monomial must be a parent of var_idx
            if not combo_set.issubset(allowed):
                mask[feat_idx, var_idx] = False

    return mask



# 3.  SINDY FIT WITH CAUSAL MASK

def fit_sindy_with_mask(
    X: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray,
    var_names: list[str],
    poly_degree: int = 1,
    threshold: float = 0.05,
    alpha: float = 0.05,
    verbose: bool = True,
) -> ps.SINDy:
    """Fit a SINDy model with a causal sparsity mask

    1. Build a polynomial feature library of the specified degree
    2. Use STLSQ (Sequential Thresholded Least Squares) as the sparse optimizer
    3. After fitting, hard-zero any remaining coefficients that violate the causal mask (catches numerical near-zeros that survived thresholding)
    4. Print discovered equations and run a self-validation mask check

    Parameters
    ----------
    X : np.ndarray, shape (T, n_vars)
        State matrix (rows = time steps, columns = variables).
    t : np.ndarray, shape (T,)
        Time vector corresponding to rows of X.
    mask : np.ndarray of bool, shape (n_feat, n_vars)
        Causal constraint mask from ``build_causal_mask()``.
    var_names : list[str]
        Human-readable variable names (same order as columns of X).
    poly_degree : int
        Polynomial library degree (default 1 = linear ODEs; use 2 for
        quadratic interactions at the cost of interpretability).
    threshold : float
        STLSQ sparsity threshold λ. Coefficients below this magnitude are
        zeroed after each least-squares step. Increase to encourage sparser
        equations; decrease to capture weaker effects.
    alpha : float
        L2 regularisation strength in STLSQ. Helps with multicollinearity.
    verbose : bool
        If True, print discovered equations and mask-check result.

    Returns
    -------
    model : ps.SINDy
        Fitted PySINDy model. Call ``model.simulate(x0, t)`` to integrate.
    """
    n_vars = X.shape[1]

    # Normalise to make threshold scale-invariant.
    # Biological variables span very different magnitudes (Activity ~8000, Mood ~7).
    # Z-scoring means 'threshold' is the same fraction of std for every variable.
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    X_std[X_std < 1e-12] = 1.0   # protect against constant columns
    X_norm = (X - X_mean) / X_std

    # Feature library
    library = ps.PolynomialLibrary(
        degree=poly_degree,
        include_bias=True,
        include_interaction=True,
    )

    # Differentiation
    # SmoothedFiniteDifference reduces noise sensitivity in bio time-series.
    differentiator = ps.SmoothedFiniteDifference(
        smoother_kws={"window_length": min(7, len(t) // 5 or 3)}
    )

    # Optimizer
    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=20)

    # Assemble and fit SINDy on normalised data
    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
        differentiation_method=differentiator,
    )

    # PySINDy v2: feature_names passed to fit(), not the constructor
    model.fit(X_norm, t=t, feature_names=var_names)

    # Enforce causal mask (hard zeroing)
    # PySINDy stores coefficients as (n_vars, n_feat). Our mask is (n_feat, n_vars).
    # Transpose mask to (n_vars, n_feat) for direct element-wise application.
    coef_matrix = model.coefficients()          # shape: (n_vars, n_feat)
    mask_T = mask.T                             # shape: (n_vars, n_feat)
    coef_matrix[~mask_T] = 0.0
    model.optimizer.coef_ = coef_matrix         # write back

    # Store normalisation stats on model for downstream use (validate, interventions)
    model._X_mean = X_mean
    model._X_std  = X_std

    # Consistency check
    if verbose:
        print("\n" + "═" * 60)
        print("  STAGE 3 (SINDy): Discovered Governing Equations")
        print("  (coefficients are in normalised / z-score space)")
        print("═" * 60)
        model.print()
        _check_mask(model, mask, var_names)

    return model


def _check_mask(model: ps.SINDy, mask: np.ndarray, var_names: list[str]) -> None:
    """Assert that all causally forbidden coefficients are zero post-fit."""
    coef = model.coefficients()   # (n_vars, n_feat)
    violations = np.sum(np.abs(coef[~mask.T]) > 1e-12)
    if violations == 0:
        print("\n  [MASK CHECK PASSED] All causally forbidden coefficients = 0.")
    else:
        print(f"\n  [MASK CHECK WARNING] {violations} coefficient(s) in forbidden "
              "positions are nonzero (numerical noise). Re-run with a higher threshold.")


# 4.  VISUALISATION

def plot_sindy_results(
    model: ps.SINDy,
    var_names: list[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot the SINDy coefficient matrix as a heatmap (mirrors Stage 4 style).

    Only the linear sub-block is shown (coefficient of x_j in the ODE for x_i),
    making it directly comparable to the VAR lag-1 heatmap from Stage 4.

    Parameters
    ----------
    model : ps.SINDy
        Fitted SINDy model.
    var_names : list[str]
        Variable names, same order as model features.
    save_path : str, optional
        If provided, saves the figure to this path instead of displaying it.
    """
    n = len(var_names)
    coef_full = model.coefficients()   # (n_vars, n_feat)

    # Build feature names list to locate linear terms
    feature_names = model.get_feature_names()

    # Extract the n×n linear sub-block (coefficient of x_j in ẋ_i equation)
    linear_coef = np.zeros((n, n))
    for j, fname in enumerate(feature_names):
        # Linear terms have exactly one variable name and no '^' or ' ' operators
        stripped = fname.strip()
        if stripped in var_names:
            col_idx = var_names.index(stripped)
            linear_coef[:, col_idx] = coef_full[:, j]

    df_coef = pd.DataFrame(linear_coef, index=var_names, columns=var_names)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: SINDy linear ODE coefficients
    sns.heatmap(
        df_coef, ax=axes[0],
        annot=True, fmt=".3f",
        cmap="RdBu_r", center=0,
        linewidths=0.5, linecolor="#e0e0e0",
        cbar_kws={"shrink": 0.85, "label": "ODE coefficient"},
    )
    axes[0].set_title(
        "Stage 3 (SINDy): Discovered ODE Coefficients\n"
        "Columns: Driver variable  |  Rows: Driven equation  ẋᵢ = Σ cᵢⱼ xⱼ",
        fontsize=10,
    )
    axes[0].set_xlabel("Driver  xⱼ", fontsize=9)
    axes[0].set_ylabel("Equation  ẋᵢ", fontsize=9)

    # Right: Sparsity pattern (causal skeleton overlay)
    sparsity = (np.abs(linear_coef) > 1e-8).astype(float)
    sns.heatmap(
        pd.DataFrame(sparsity, index=var_names, columns=var_names),
        ax=axes[1],
        annot=True, fmt=".0f",
        cmap="Greens",
        linewidths=0.5, linecolor="#e0e0e0",
        cbar_kws={"shrink": 0.85, "label": "Active edge (1=yes)"},
        vmin=0, vmax=1,
    )
    axes[1].set_title(
        "Causal Sparsity Pattern\n"
        "1 = causally active edge (from Stage 2 PC + SINDy sparsity)",
        fontsize=10,
    )
    axes[1].set_xlabel("Driver  xⱼ", fontsize=9)
    axes[1].set_ylabel("Equation  ẋᵢ", fontsize=9)

    plt.suptitle("SOMA — SINDy Mechanistic ODE Discovery", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Figure saved → {save_path}")
    else:
        plt.show()


# 5.  ODE INTEGRATION & VALIDATION

def validate_sindy_fit(
    model: ps.SINDy,
    X: np.ndarray,
    t: np.ndarray,
    var_names: list[str],
    save_path: Optional[str] = None,
) -> dict[str, float]:
    """Integrate the discovered ODEs and compare to observed data.

    Uses ``scipy.integrate.solve_ivp`` (RK45) starting from the first
    observed state X[0] and integrates over the same time span as the data.

    Parameters
    ----------
    model : ps.SINDy
        Fitted SINDy model.
    X : np.ndarray, shape (T, n_vars)
        Observed state matrix.
    t : np.ndarray, shape (T,)
        Time vector.
    var_names : list[str]
        Variable names.
    save_path : str, optional
        Save figure to this path rather than displaying.

    Returns
    -------
    rmse_per_var : dict[str, float]
        RMSE of simulated vs. observed trajectory for each variable.
    """
    n_vars = X.shape[1]

    # Retrieve normalisation stats stored during fit
    X_mean = getattr(model, "_X_mean", np.zeros(n_vars))
    X_std  = getattr(model, "_X_std",  np.ones(n_vars))

    # Normalise to the same space the model was fitted in
    X_norm = (X - X_mean) / X_std
    x0_norm = X_norm[0, :]

    # Integrate in normalised space
    def ode_rhs(t_scalar: float, x: np.ndarray) -> np.ndarray:
        # model.predict expects shape (1, n_vars)
        return model.predict(x.reshape(1, -1)).flatten()

    sol = solve_ivp(
        ode_rhs,
        t_span=(t[0], t[-1]),
        y0=x0_norm,
        t_eval=t,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        print(f"  [WARNING] ODE integration failed: {sol.message}")
        return {}

    # Denormalise back to original units
    X_sim_norm = sol.y.T                         # (T, n_vars) in z-score space
    X_sim = X_sim_norm * X_std + X_mean          # back to original units

    # RMSE in original units
    rmse_per_var: dict[str, float] = {}
    for i, name in enumerate(var_names):
        rmse = float(np.sqrt(np.mean((X_sim[:, i] - X[:, i]) ** 2)))
        rmse_per_var[name] = rmse

    print("\n  SINDy ODE Validation — RMSE (simulated vs. observed):")
    for name, rmse in rmse_per_var.items():
        print(f"    {name:12s}  RMSE = {rmse:.4f}")

    # Plot
    n_cols = 2
    n_rows = int(np.ceil(n_vars / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.5 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, name in enumerate(var_names):
        ax = axes[i]
        ax.plot(t, X[:, i], color="#1a1a2e", lw=1.8, label="Observed", alpha=0.85)
        ax.plot(t, X_sim[:, i], color="#e94560", lw=1.5,
                linestyle="--", label="SINDy (integrated)", alpha=0.9)
        ax.set_title(f"{name}  (RMSE={rmse_per_var[name]:.3f})", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
        ax.set_xlabel("Time", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Stage 3 (SINDy): Simulated vs. Observed Trajectories", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")
    else:
        plt.show()

    return rmse_per_var


# 6.  EQUATION EXTRACTOR  (utility for downstream scripts)

def extract_equations(model: ps.SINDy, var_names: list[str]) -> dict[str, str]:
    """Return the discovered ODE equations as a human-readable dictionary.

    Returns
    -------
    equations : dict[str, str]
        Mapping  variable_name → equation string, e.g.
        ``{'Glucose': 'ẋ_Glucose = 18.24 + -0.031 Activity + 0.18 Glucose'}``
    """
    feature_names = model.get_feature_names()
    coef = model.coefficients()   # (n_vars, n_feat)
    equations: dict[str, str] = {}

    for i, var in enumerate(var_names):
        terms = []
        for j, feat in enumerate(feature_names):
            c = coef[i, j]
            if abs(c) > 1e-12:
                terms.append(f"{c:+.4f} {feat}" if feat != "1" else f"{c:+.4f}")
        rhs = "  ".join(terms) if terms else "0"
        equations[var] = f"ẋ_{var} = {rhs}"

    return equations


# 7.  STANDALONE SMOKE TEST 
#     Uses the same synthetic data generator as deep-learning.py so no
#     external file is needed

def _generate_synthetic_data(n_days: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    """same as deep-learning.py"""
    np.random.seed(42)
    var_names = ["Sleep", "Mood", "Activity", "RHR", "HRV", "VO2_Max", "Glucose"]
    data = np.zeros((n_days, 7))
    data[0] = [75, 7, 5000, 60, 50, 45, 90]

    for t in range(1, n_days):
        p = data[t - 1]
        p_sleep, p_mood, p_act, p_rhr, p_hrv, p_vo2, p_gluc = p
        new_sleep  = np.clip(0.3 * p_sleep + 50 + np.random.normal(0, 8), 0, 100)
        new_hrv    = 0.4 * p_hrv + 0.3 * (p_sleep - 50) + 2 * p_mood + np.random.normal(0, 5)
        new_rhr    = 0.6 * p_rhr - 0.1 * (new_hrv - 50) - 0.1 * (p_sleep - 70) + np.random.normal(0, 2)
        new_mood   = np.clip(0.4 * p_mood + 0.05 * p_sleep + 0.02 * p_hrv + np.random.normal(0, 1), 1, 10)
        new_act    = 0.5 * p_act + 500 * p_mood + 50 * p_vo2 + np.random.normal(0, 1000)
        new_vo2    = 0.95 * p_vo2 + 0.0001 * p_act + np.random.normal(0, 0.2)
        new_gluc   = 0.7 * p_gluc - 0.001 * p_act - 0.1 * (p_sleep - 70) + 30 + np.random.normal(0, 3)
        data[t]    = [new_sleep, new_mood, new_act, new_rhr, new_hrv, new_vo2, new_gluc]

    df = pd.DataFrame(data, columns=var_names)

    # Fake PC adjacency:
    #   sleep → hrv, sleep → rhr, sleep → mood, sleep → glucose
    #   mood  → hrv, mood → activity
    #   hrv   → rhr
    #   activity → vo2, activity → glucose
    #   vo2   → activity (feedback)
    n = len(var_names)
    idx = {v: i for i, v in enumerate(var_names)}
    adj = np.zeros((n, n), dtype=int)

    def add_edge(u: str, v: str):
        i, j = idx[u], idx[v]
        adj[j, i] = 1    # arrow points TO j
        adj[i, j] = -1   # tail FROM i

    edges = [
        ("Sleep", "HRV"), ("Sleep", "RHR"), ("Sleep", "Mood"), ("Sleep", "Glucose"),
        ("Mood", "HRV"), ("Mood", "Activity"),
        ("HRV", "RHR"),
        ("Activity", "VO2_Max"), ("Activity", "Glucose"),
        ("VO2_Max", "Activity"),
    ]
    for u, v in edges:
        add_edge(u, v)

    return df, adj


def _run_smoke_test():
    print("=" * 65)
    print("  SOMA — SINDy Stage 3 Smoke Test (Synthetic Data)")
    print("=" * 65)

    # Generate data & fake PC matrix
    df_train, adj_fake = _generate_synthetic_data(n_days=300)
    var_names = list(df_train.columns)
    print(f"\n  Data shape: {df_train.shape}   Variables: {var_names}")

    # DataAdapter
    adapter = DataAdapter(df_train, dt=1.0)
    X, t = adapter.X, adapter.t

    # Causal mask
    mask = build_causal_mask(
        adj=adj_fake,
        var_names=var_names,
        poly_degree=1,
        include_bias=True,
    )
    print(f"\n  Causal mask shape: {mask.shape}  "
          f"(n_features={mask.shape[0]}, n_vars={mask.shape[1]})")
    print(f"  Allowed coefficients: {mask.sum()} / {mask.size}"
          f"  ({100*mask.mean():.1f}% non-zero structure)")

    # Fit SINDy
    model = fit_sindy_with_mask(
        X=X, t=t,
        mask=mask,
        var_names=var_names,
        poly_degree=1,
        threshold=0.02,
        alpha=0.01,
        verbose=True,
    )

    # Human-readable equations
    print("\n  Extracted ODE equations:")
    eqs = extract_equations(model, var_names)
    for var, eq in eqs.items():
        print(f"    {eq}")

    # Visualise coefficient heatmap
    print("\n  Generating coefficient heatmap…")
    plot_sindy_results(model, var_names)

    # Validate: integrate & compare
    print("\n  Validating ODE integration vs. observed data…")
    rmse = validate_sindy_fit(model, X, t, var_names)

    print("\n" + "=" * 65)
    print("  Smoke test complete.  ✓")
    print("=" * 65)
    return model, mask, rmse



# ENTRY POINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SINDy Stage 3 — Mechanistic ODE Discovery for SOMA"
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "cohort"],
        default="synthetic",
        help="Data source: 'synthetic' (built-in generator) or 'cohort' "
             "(requires cohort_study_5000.xlsx and prior Stage 2 run).",
    )
    args = parser.parse_args()

    if args.mode == "synthetic":
        _run_smoke_test()
    else:
        print(
            textwrap.dedent("""
            Cohort mode: import sindy_stage3 into cohorstudy.py after Stage 2.

            Example integration snippet
            ───────────────────────────
            from sindy_stage3 import DataAdapter, build_causal_mask, \\
                                     fit_sindy_with_mask, plot_sindy_results, \\
                                     validate_sindy_fit

            # After the PC block (Stage 2) in cohorstudy.py:
            adapter = DataAdapter(df_train_ts, dt=7.0)   # weekly = dt=7
            mask    = build_causal_mask(adj, labels, poly_degree=1)
            model   = fit_sindy_with_mask(adapter.X, adapter.t, mask, labels)
            plot_sindy_results(model, labels)
            rmse    = validate_sindy_fit(model, adapter.X, adapter.t, labels)
            """)
        )
