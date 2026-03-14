"""
sindy_weather.py — SINDy Stage 3 integration for the weather pipeline.

Plugs into weatherdata.py after Stage 2 (PC Algorithm).
Run after weatherdata.py has been executed, or call the functions
directly by importing this module.

Key differences from the synthetic bio-data case:
  • 5 variables: TMAX, TMIN, PRCP, SNOW, SNWD  (daily, deg C / mm / mm)
  • Data is spatially-aggregated daily means → much smoother than per-station
  • Strong seasonal cycle in TMAX/TMIN → must be removed before SINDy so
    the model captures inter-variable physics, not just the calendar
  • Physical laws are genuinely continuous: snowmelt driven by temperature
    is  dSNWD/dt = f(TMAX, SNWD) — exactly what SINDy is designed for

Usage (standalone):
    python sindy_weather.py

Usage (after running weatherdata.py in the same process):
    from sindy_weather import run_sindy_weather
    run_sindy_weather(df_train_ts, df_test_ts, adj, weather_vars)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import savgol_filter
from typing import Optional

try:
    import pysindy as ps
except ImportError:
    raise ImportError("PySINDy required: pip install pysindy")

# Import the core SINDy helpers from sindy.py (must be in the same directory)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sindy import (
    DataAdapter,
    build_causal_mask,
    fit_sindy_with_mask,
    plot_sindy_results,
    validate_sindy_fit,
    extract_equations,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SEASONAL DETRENDING
#     Remove the annual sinusoidal cycle before SINDy so it captures
#     inter-variable physics, not just "summer is warm, winter is cold."
# ─────────────────────────────────────────────────────────────────────────────

def detrend_seasonal(
    df: pd.DataFrame,
    window: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove slow seasonal trend via Savitzky-Golay smoothing.

    Strategy
    ────────
    1. Fit a 30-day Savitzky-Golay filter (polynomial order 2) to each
       variable → this captures the seasonal envelope.
    2. Subtract it → residuals contain the day-to-day physics (storm
       events, cold snaps, snowmelt pulses) that SINDy should identify.
    3. Also apply a 3-day rolling mean to the residuals to further
       reduce sensor noise before automatic differentiation.

    Returns
    ───────
    df_resid : pd.DataFrame   — detrended, lightly smoothed data (feed to SINDy)
    df_trend : pd.DataFrame   — seasonal component (for plotting)
    """
    df_trend  = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    df_resid  = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        series = df[col].to_numpy(dtype=float)
        # Savitzky-Golay requires window < len(data) and window must be odd
        wl = min(window | 1, len(series) - 1)   # ensure odd, ≤ len-1
        if wl < 5:
            wl = 5
        trend = savgol_filter(series, window_length=wl, polyorder=2)
        df_trend[col]  = trend
        # Light 3-day smoother on residuals to reduce noise
        resid_raw      = series - trend
        df_resid[col]  = pd.Series(resid_raw).rolling(3, center=True,
                                                       min_periods=1).mean().values

    return df_resid, df_trend


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONTINUOUS-TIME WEATHER COUNTERFACTUALS
# ─────────────────────────────────────────────────────────────────────────────

def weather_counterfactual(
    model: ps.SINDy,
    x0: np.ndarray,
    t_eval: np.ndarray,
    var_names: list[str],
    intervention_var: str,
    delta: float,
    scenario_label: str,
    save_path: Optional[str] = None,
) -> None:
    """Continuous-time weather counterfactual: What if TMAX is +2°C warmer?

    Parameters
    ----------
    model : ps.SINDy
        Fitted SINDy model (in normalised/detrended space).
    x0 : np.ndarray
        Initial state (normalised).
    t_eval : np.ndarray
        Integration time grid (days).
    var_names : list[str]
        Variable names.
    intervention_var : str
        Variable to perturb.
    delta : float
        Constant offset to add to the intervention variable at every step
        (in the same z-score space as the fitted model).
    scenario_label : str
        Human-readable scenario description.
    save_path : str, optional
        Save figure here instead of displaying.
    """
    from scipy.integrate import solve_ivp

    X_mean = getattr(model, "_X_mean", np.zeros(len(var_names)))
    X_std  = getattr(model, "_X_std",  np.ones(len(var_names)))
    iv_idx = var_names.index(intervention_var)

    def rhs_baseline(t, x):
        return model.predict(x.reshape(1, -1)).flatten()

    def rhs_intervention(t, x):
        x_mod = x.copy()
        x_mod[iv_idx] += delta           # constant perturbation in z-score space
        return model.predict(x_mod.reshape(1, -1)).flatten()

    sol_base = solve_ivp(rhs_baseline,    (t_eval[0], t_eval[-1]),
                         x0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)
    sol_int  = solve_ivp(rhs_intervention,(t_eval[0], t_eval[-1]),
                         x0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)

    # Denormalise
    X_base = sol_base.y.T * X_std + X_mean
    X_int  = sol_int.y.T  * X_std + X_mean

    show_vars = [v for v in var_names if v != intervention_var]
    n = len(show_vars)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    colors = {"base": "#264653", "int": "#e76f51"}
    for ax, vname in zip(axes, show_vars):
        vi = var_names.index(vname)
        ax.plot(t_eval, X_base[:, vi], color=colors["base"],
                lw=2, label="Baseline", alpha=0.85)
        ax.plot(t_eval, X_int[:, vi],  color=colors["int"],
                lw=2, linestyle="--", label=scenario_label, alpha=0.9)
        ax.fill_between(t_eval, X_base[:, vi], X_int[:, vi],
                        alpha=0.15, color="#2a9d8f")
        div = X_int[-1, vi] - X_base[-1, vi]
        ax.set_title(f"{vname}  (Δ={div:+.3f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Days", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    plt.suptitle(
        f"Weather Counterfactual: {scenario_label}\n"
        f"(in detrended / anomaly space)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_sindy_weather(
    df_train_ts: pd.DataFrame,
    df_test_ts: pd.DataFrame,
    adj: np.ndarray,
    weather_vars: list[str],
    poly_degree: int = 1,
    threshold: float = 0.05,
    seasonal_window: int = 30,
) -> ps.SINDy:
    """Run the full SINDy pipeline on aggregated weather time-series.

    Parameters
    ----------
    df_train_ts : pd.DataFrame
        Daily aggregated train time-series (datetime index, columns = weather_vars).
    df_test_ts : pd.DataFrame
        Daily aggregated test time-series.
    adj : np.ndarray
        PC CPDAG adjacency matrix (cg.G.graph).
    weather_vars : list[str]
        Variable names (e.g. ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'SNWD']).
    poly_degree : int
        SINDy library polynomial degree.
    threshold : float
        STLSQ sparsity threshold (in z-score space).
    seasonal_window : int
        Savitzky-Golay window for seasonal detrending (days).

    Returns
    -------
    model : ps.SINDy
        Fitted SINDy model (in detrended z-score space).
    """
    print("=" * 65)
    print("  STAGE 3 (SINDy) — Weather Data")
    print("=" * 65)

    # ── 3a. Seasonal detrending ──────────────────────────────────────────────
    print(f"\n  Detrending seasonal cycle (Savitzky-Golay, window={seasonal_window}d)…")
    df_train_detrend, df_train_trend = detrend_seasonal(df_train_ts[weather_vars],
                                                        window=seasonal_window)
    df_test_detrend, _               = detrend_seasonal(df_test_ts[weather_vars],
                                                        window=seasonal_window)

    # Quick sanity plot: raw vs. detrended for TMAX
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(df_train_ts.index, df_train_ts["TMAX"], lw=0.8, color="#264653")
    axes[0].plot(df_train_ts.index, df_train_trend["TMAX"],
                 lw=2, color="#e76f51", label="Seasonal trend")
    axes[0].set_ylabel("TMAX (raw)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

    axes[1].plot(df_train_ts.index, df_train_detrend["TMAX"],
                 lw=0.8, color="#2a9d8f", label="Detrended (anomaly)")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_ylabel("TMAX anomaly"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    plt.suptitle("Seasonal Detrending: TMAX", fontsize=11)
    plt.tight_layout(); plt.show()

    # ── 3b. Adapt to SINDy format ────────────────────────────────────────────
    adapter = DataAdapter(df_train_detrend, dt=1.0)   # daily = dt=1
    X, t = adapter.X, adapter.t
    print(f"  Training data: {X.shape[0]} days × {X.shape[1]} variables")

    # ── 3c. Causal mask from PC output ───────────────────────────────────────
    mask = build_causal_mask(adj, weather_vars, poly_degree=poly_degree)
    print(f"  Causal mask: {mask.sum()}/{mask.size} coefficients allowed "
          f"({100*mask.mean():.0f}% non-zero structure)")

    # ── 3d. Fit SINDy ────────────────────────────────────────────────────────
    model = fit_sindy_with_mask(
        X=X, t=t,
        mask=mask,
        var_names=weather_vars,
        poly_degree=poly_degree,
        threshold=threshold,
        alpha=0.01,
        verbose=True,
    )

    # ── 3e. Coefficient heatmap ───────────────────────────────────────────────
    print("\n  Generating ODE coefficient heatmap…")
    plot_sindy_results(model, weather_vars)

    # ── 3f. Human-readable equations ─────────────────────────────────────────
    print("\n  Discovered governing equations (in detrended / anomaly space):")
    eqs = extract_equations(model, weather_vars)
    for var, eq in eqs.items():
        print(f"    {eq}")

    # ── 3g. ODE integration validation on TRAIN ──────────────────────────────
    print("\n  Validating on TRAIN (integrated ODE vs. observed anomaly)…")
    validate_sindy_fit(model, X, t, weather_vars)

    # ── 3h. Counterfactual: "What if TMAX anomaly is persistently +1 std?" ───
    # In z-score space, delta=1.0 corresponds to +1 standard deviation of TMAX
    print("\n  Running counterfactual: TMAX anomaly +1 std persistently…")
    X_norm = (X - model._X_mean) / model._X_std
    x0_norm = X_norm[0, :]
    t_cf = np.linspace(0, 90, 300)   # 90-day horizon

    weather_counterfactual(
        model=model,
        x0=x0_norm,
        t_eval=t_cf,
        var_names=weather_vars,
        intervention_var="TMAX",
        delta=1.0,     # +1 std in z-score space ≈ +1 × σ_TMAX degrees warmer
        scenario_label="TMAX +1σ (persistent warming)",
    )

    print("\n" + "=" * 65)
    print("  SINDy Weather Stage Complete  ✓")
    print("=" * 65)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    DATA_PATH = "/Users/isabellamueller-vogt/Library/Mobile Documents/com~apple~CloudDocs/08 - side quests/network-dynamics-failure-propagation/model/data/ghcn_clean_small.csv"

    print("Loading GHCN weather data…")
    df_long = pd.read_csv(DATA_PATH, parse_dates=["date"])

    weather_vars = ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]
    df_long = df_long[df_long["variable"].isin(weather_vars)].copy()

    df_wide = (
        df_long
        .pivot_table(index=["station", "date"], columns="variable", values="value")
        .reset_index()
        .dropna(subset=weather_vars)
    )

    # Train/test split by station (same as weatherdata.py)
    from sklearn.model_selection import train_test_split
    stations = df_wide["station"].unique()
    train_stn, test_stn = train_test_split(stations, test_size=0.2, random_state=42)

    train_data = df_wide[df_wide["station"].isin(train_stn)]
    test_data  = df_wide[df_wide["station"].isin(test_stn)]

    df_train_ts = train_data.groupby("date")[weather_vars].mean().sort_index()
    df_test_ts  = test_data.groupby("date")[weather_vars].mean().sort_index()

    print(f"Train: {df_train_ts.shape}   Test: {df_test_ts.shape}")
    print(f"Date range: {df_train_ts.index.min()} → {df_train_ts.index.max()}")

    # Stage 2: PC Algorithm
    from causallearn.search.ConstraintBased.PC import pc
    df_train_pc = train_data[weather_vars].dropna()
    cg = pc(df_train_pc.to_numpy(), alpha=0.05, indep_test="fisherz", verbose=False)
    adj = cg.G.graph
    print("\nPC Algorithm complete. Adjacency matrix shape:", adj.shape)

    # Stage 3: SINDy (this script)
    run_sindy_weather(df_train_ts, df_test_ts, adj, weather_vars)
