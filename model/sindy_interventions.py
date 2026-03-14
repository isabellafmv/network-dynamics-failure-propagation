"""Continuous-time counterfactual simulations for SOMA.

-> mechanistic, continuous-time trajectory comparisons using the ODE system (from sindy)

Core idea
---------
An "intervention" is a callable  f(t, x) → x_modified  that clamps or
offsets one or more variables at every integration step.  By running the same
ODE with and without this intervention we get causal counterfactuals:

    "What is the long-term trajectory of Glucose if Sleep is
     permanently increased by 20%?"

Usage (standalone demo):
    python sindy_interventions.py

Usage (import into an existing pipeline after sindy_stage3.py):
    from sindy_interventions import simulate_counterfactual, \
                                    compare_baseline_vs_intervention, \
                                    InterventionFactory
"""

from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

try:
    import pysindy as ps
except ImportError as e:
    raise ImportError("PySINDy is required: pip install pysindy") from e


# 1.  INTERVENTION FACTORY
#     Convenience constructors for the most common biological interventions.

class InterventionFactory:
    """factory methods that return intervention callables

    Each returned callable has the signature:
        intervention(t: float, x: np.ndarray) -> np.ndarray

    where x is the state vector at time t.  The callable should return a *modified* copy of x with the intervention applied.
    """

    @staticmethod
    def constant_increase(
        var_idx: int,
        relative_increase: float = 0.20,
        baseline_value: float = 0.0,
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """Permanently clamp a variable to (1 + relative_increase) x baseline.

        Parameters
        ----------
        var_idx : int
            Index of the variable to intervene on.
        relative_increase : float
            Fractional increase, e.g. 0.20 = +20 %.
        baseline_value : float
            The unperturbed value to scale from (typically X[0, var_idx]).
        """
        target = baseline_value * (1.0 + relative_increase)

        def _intervention(t: float, x: np.ndarray) -> np.ndarray:
            x = x.copy()
            x[var_idx] = target
            return x

        _intervention.__doc__ = (
            f"Clamp variable[{var_idx}] to {target:.3f} "
            f"(+{100*relative_increase:.0f}% of baseline={baseline_value:.3f})"
        )
        return _intervention

    @staticmethod
    def step_change(
        var_idx: int,
        delta: float,
        t_start: float = 0.0,
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """Add a constant offset to a variable after time t_start.

        Parameters
        ----------
        var_idx : int
            Index of the variable to perturb.
        delta : float
            Absolute offset applied at every time step ≥ t_start.
        t_start : float
            Time at which the intervention begins (useful for delayed onset).
        """
        def _intervention(t: float, x: np.ndarray) -> np.ndarray:
            if t < t_start:
                return x
            x = x.copy()
            x[var_idx] += delta
            return x

        return _intervention

    @staticmethod
    def periodic_boost(
        var_idx: int,
        amplitude: float,
        period: float,
        phase: float = 0.0,
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """Apply a sinusoidal periodic boost to a variable (e.g. weekly exercise).

        Models interventions that are not constant but oscillate — for example
        a workout regimen that increases VO2_Max by up to ``amplitude`` units
        with a cycle of ``period`` days.
        """
        def _intervention(t: float, x: np.ndarray) -> np.ndarray:
            x = x.copy()
            x[var_idx] += amplitude * np.sin(2 * np.pi * (t - phase) / period)
            return x

        return _intervention


# 2.  CORE SIMULATION ENGINE

def simulate_counterfactual(
    model: ps.SINDy,
    x0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    intervention_fn: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> np.ndarray:
    """Integrate the SINDy ODE, optionally applying a continuous intervention.

    Parameters
    ----------
    model : ps.SINDy
        Fitted SINDy model (from ``sindy_stage3.fit_sindy_with_mask``).
    x0 : np.ndarray, shape (n_vars,)
        Initial state.
    t_span : tuple[float, float]
        (t_start, t_end) integration interval in natural time units.
    t_eval : np.ndarray
        Dense time grid at which to record the solution.
    intervention_fn : callable, optional
        ``f(t, x) → x_modified``.  Applied to the state *before* evaluating
        the ODE RHS at each step.  ``None`` = baseline (no intervention).
    rtol, atol : float
        Tolerances for the RK45 integrator.

    Returns
    -------
    X_sim : np.ndarray, shape (len(t_eval), n_vars)
        Simulated state trajectory.
    """

    def rhs(t_scalar: float, x: np.ndarray) -> np.ndarray:
        if intervention_fn is not None:
            x = intervention_fn(t_scalar, x)
        return model.predict(x.reshape(1, -1)).flatten()

    sol = solve_ivp(
        rhs,
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    return sol.y.T  # (T, n_vars)


# 3.  COMPARISON PLOT

def compare_baseline_vs_intervention(
    model: ps.SINDy,
    x0: np.ndarray,
    t_eval: np.ndarray,
    intervention_fn: Callable[[float, np.ndarray], np.ndarray],
    var_names: list[str],
    response_vars: Optional[list[str]] = None,
    scenario_label: str = "Intervention",
    save_path: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """Simulate baseline and intervention trajectories and plot the comparison.

    Parameters
    ----------
    model : ps.SINDy
        Fitted SINDy model.
    x0 : np.ndarray, shape (n_vars,)
        Shared initial state for both simulations.
    t_eval : np.ndarray, shape (T,)
        Evaluation time grid.
    intervention_fn : callable
        Intervention callable, e.g. from ``InterventionFactory``.
    var_names : list[str]
        All variable names.
    response_vars : list[str], optional
        Subset of variables to show in the comparison plot.  If ``None``,
        show all variables.
    scenario_label : str
        Human-readable description shown in plot legend and title.
    save_path : str, optional
        Save figure to this path instead of displaying.

    Returns
    -------
    results : dict
        ``{'baseline': X_base, 'intervention': X_int, 'divergence': X_int - X_base}``
    """
    t_span = (float(t_eval[0]), float(t_eval[-1]))

    # ── 3a. Run simulations ───────────────────────────────────────────────────
    print(f"  Simulating baseline trajectory…")
    X_base = simulate_counterfactual(model, x0, t_span, t_eval)

    print(f"  Simulating counterfactual: {scenario_label}…")
    X_int  = simulate_counterfactual(model, x0, t_span, t_eval, intervention_fn)

    divergence = X_int - X_base

    # ── 3b. Select variables to plot ──────────────────────────────────────────
    if response_vars is None:
        response_vars = var_names
    plot_idx = [var_names.index(v) for v in response_vars]

    n_plots = len(plot_idx)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.8 * n_rows))
    axes = np.array(axes).flatten()

    colors = {"baseline": "#264653", "intervention": "#e76f51", "divergence": "#2a9d8f"}

    for plot_pos, var_idx in enumerate(plot_idx):
        ax = axes[plot_pos]
        vname = var_names[var_idx]

        ax.plot(t_eval, X_base[:, var_idx],
                color=colors["baseline"], lw=2, label="Baseline", alpha=0.85)
        ax.plot(t_eval, X_int[:, var_idx],
                color=colors["intervention"], lw=2, linestyle="--",
                label=scenario_label, alpha=0.9)

        # Shaded divergence region
        ax.fill_between(
            t_eval,
            X_base[:, var_idx],
            X_int[:, var_idx],
            alpha=0.15,
            color=colors["divergence"],
            label="Divergence",
        )

        # Annotate terminal divergence
        terminal_div = divergence[-1, var_idx]
        sign = "↑" if terminal_div > 0 else "↓"
        ax.annotate(
            f"  Δ={terminal_div:+.2f} {sign}",
            xy=(t_eval[-1], X_int[-1, var_idx]),
            fontsize=8, color=colors["intervention"],
            ha="left", va="center",
        )

        ax.set_title(vname, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    for j in range(plot_pos + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"SOMA — Continuous-Time Counterfactual\n"
        f"Scenario: {scenario_label}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")
    else:
        plt.show()

    return {"baseline": X_base, "intervention": X_int, "divergence": divergence}


# 4.  THRESHOLD DETECTION
#     Identify when divergence between scenarios crosses a clinically meaningful bound

def find_intervention_thresholds(
    results: dict[str, np.ndarray],
    t_eval: np.ndarray,
    var_names: list[str],
    threshold_pct: float = 5.0,
) -> dict[str, Optional[float]]:
    """Find the earliest time where the counterfactual diverges meaningfully.

    A variable is considered to have crossed its "Acute Intervention Threshold"
    when the absolute divergence exceeds threshold_pct % of the baseline
    value at that time.

    Parameters
    ----------
    results : dict
        Output of ``compare_baseline_vs_intervention``.
    t_eval : np.ndarray
        Evaluation time grid.
    var_names : list[str]
        Variable names.
    threshold_pct : float
        Divergence threshold as a percentage of the baseline value.

    Returns
    -------
    thresholds : dict[str, Optional[float]]
        Variable → time at which the threshold is first crossed,
        or None if the threshold is never reached.
    """
    X_base    = results["baseline"]
    divergence = results["divergence"]
    thresholds: dict[str, Optional[float]] = {}

    print(f"\n  Acute Intervention Thresholds (>{threshold_pct:.0f}% divergence):")
    for i, vname in enumerate(var_names):
        baseline_vals = X_base[:, i]
        abs_div = np.abs(divergence[:, i])
        # Relative divergence at each time step
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_div = np.where(np.abs(baseline_vals) > 1e-8,
                               abs_div / np.abs(baseline_vals) * 100.0,
                               0.0)
        crossings = np.where(rel_div >= threshold_pct)[0]
        if len(crossings) > 0:
            t_cross = t_eval[crossings[0]]
            thresholds[vname] = float(t_cross)
            print(f"    {vname:12s}  threshold crossed at t={t_cross:.1f}  "
                  f"(div={rel_div[crossings[0]]:.1f}%)")
        else:
            thresholds[vname] = None
            print(f"    {vname:12s}  threshold NOT reached within simulation window")

    return thresholds


# 5.  STANDALONE DEMO

def _run_demo():
    """Run a self-contained demo using the sindy_stage3 smoke-test model."""
    print("=" * 65)
    print("  SOMA — Stage 5 Continuous-Time Counterfactual Demo")
    print("=" * 65)

    # Re-use the smoke-test from sindy_stage3 
    try:
        from sindy_stage3 import _run_smoke_test
    except ImportError:
        raise ImportError(
            "sindy_stage3.py must be in the same directory. "
            "Run:  python sindy_stage3.py  first."
        )

    model, mask, _ = _run_smoke_test()

    # Define initial conditions
    var_names = ["Sleep", "Mood", "Activity", "RHR", "HRV", "VO2_Max", "Glucose"]
    # Physiologically plausible starting point
    x0 = np.array([75.0, 7.0, 5000.0, 60.0, 50.0, 45.0, 90.0])

    t_eval = np.linspace(0, 90, 500)   # 90-day horizon, 500 points

    # Scenario 1: "What if Sleep is permanently increased by 20%?"
    # Expected: Glucose should decrease (lower fasting glucose), HRV should
    # rise, RHR should fall.
    sleep_idx = var_names.index("Sleep")
    sleep_baseline = float(x0[sleep_idx])

    sleep_intervention = InterventionFactory.constant_increase(
        var_idx=sleep_idx,
        relative_increase=0.20,
        baseline_value=sleep_baseline,
    )

    print("\n  Scenario 1: Sleep → +20% permanently")
    results_1 = compare_baseline_vs_intervention(
        model=model,
        x0=x0,
        t_eval=t_eval,
        intervention_fn=sleep_intervention,
        var_names=var_names,
        response_vars=["Glucose", "RHR", "HRV", "Mood", "Activity"],
        scenario_label="Sleep +20%",
    )

    thresholds_1 = find_intervention_thresholds(
        results_1, t_eval, var_names, threshold_pct=3.0
    )

    # Scenario 2: "What if VO2_Max drops by 15% (cardiorespiratory decline)?"
    # Models a failure propagation scenario.
    vo2_idx = var_names.index("VO2_Max")
    vo2_baseline = float(x0[vo2_idx])

    decline_intervention = InterventionFactory.constant_increase(
        var_idx=vo2_idx,
        relative_increase=-0.15,
        baseline_value=vo2_baseline,
    )

    print("\n  Scenario 2: VO2_Max decline by 15% (failure propagation)")
    results_2 = compare_baseline_vs_intervention(
        model=model,
        x0=x0,
        t_eval=t_eval,
        intervention_fn=decline_intervention,
        var_names=var_names,
        response_vars=["Activity", "RHR", "Glucose", "HRV"],
        scenario_label="VO2_Max −15% (decline)",
    )

    thresholds_2 = find_intervention_thresholds(
        results_2, t_eval, var_names, threshold_pct=3.0
    )

    print("\n" + "=" * 65)
    print("  Counterfactual demo complete.  ✓")
    print("=" * 65)


if __name__ == "__main__":
    _run_demo()
