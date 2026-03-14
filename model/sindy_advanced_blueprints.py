"""
sindy_advanced_blueprints.py -> Bayesian Inversion & Hybrid Neural ODE

 A. Bayesian Parameter Estimation (PyMC)
    Move from single-point ODE coefficient estimates to probability
    distributions of causal strengths.

 B. Hybrid Neural ODE (PyTorch + torchdiffeq)
    SINDy equations form the "physics prior"; a small residual MLP learns
    what SINDy cannot explain (missed nonlinearities, measurement noise,
    unmeasured confounders / failure modes).

How to read this file
─────────────────────
Each blueprint is wrapped in a function with an explicit list of
prerequisites and a step-by-step prose explanation.  The code inside is
runnable scaffolding — it will execute once dependencies are installed,
but is designed to be read and extended rather than used as-is for a
production system.
"""

from __future__ import annotations

# A.  BAYESIAN PARAMETER ESTIMATION WITH PyMC

def bayesian_ode_estimation(
    sindy_model,
    X_observed: "np.ndarray",
    t: "np.ndarray",
    var_names: list[str],
    n_samples: int = 1000,
    n_tune: int = 500,
    target_accept: float = 0.9,
):
    """
    Bayesian estimation of ODE parameters using PyMC.

    Motivation
    ──────────
    SINDy gives you *point estimates* of ODE coefficients.  In health contexts you often need uncertainty:

      "Activity has a coefficient of -0.031 on Glucose — but is the 95% credible interval entirely negative?"

    PyMC solves this by placing prior distributions on each coefficient,
    then sampling the posterior given the observed data.

    Architecture
    ────────────
    1. Prior:
         θᵢⱼ ~ Normal(μ=SINDy_coef[i,j], σ=1.0)
       (centred on the SINDy estimate; wider σ = less informative).

    2. Likelihood:
         For each time in t[1:], approximate dX/dt via finite differences
         (Δx / Δt) and compare to Θ(X) · θ (the ODE RHS given the current
         parameters).  Residuals are modelled as iid Gaussian noise.

         dX_obs[t] ~ Normal(Θ(X[t]) · θ,  σ_noise²)

       This "collocation" approach avoids the expensive inner ODE solve at
       each MCMC step and is standard in mechanistic Bayesian modelling.

    3. Posterior:
         Sampled with NUTS (No-U-Turn Sampler) — the gold standard for
         continuous parameter spaces.

    Parameters
    ──────────
    sindy_model  : ps.SINDy
        Fitted SINDy model (provides coefficient prior means and the
        feature library Θ).
    X_observed   : np.ndarray, shape (T, n_vars)
        Observed state matrix.
    t            : np.ndarray, shape (T,)
        Time vector.
    var_names    : list[str]
        Variable names.
    n_samples    : int
        Number of posterior samples per chain.
    n_tune       : int
        Number of tuning (warm-up) steps.
    target_accept : float
        Target acceptance rate for NUTS (0.9 is robust default).

    Returns
    ───────
    trace : az.InferenceData
        Posterior samples in ArviZ format.  Use az.summary(trace) and
        az.plot_posterior(trace) to inspect results.

    Next steps after sampling
    ────────────────────────
    • Plot credible intervals on trajectories: integrate ODE for each
      posterior draw, plot 5th/95th percentile band.
    • Identify significant edges: variables whose posterior 95% CI
      excludes zero are "credibly non-zero" effects.
    • Compare priors vs. posteriors to see how much the data updates the
      SINDy point estimate.
    """
    import numpy as np

    try:
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError:
        raise ImportError(
            "Bayesian blueprint requires PyMC:\n"
            "  pip install pymc pytensor"
        )

    # ── A1. Extract SINDy feature library on observed data ──────────────────
    #   Θ has shape (T, n_feat).  We'll use it as the design matrix.
    import pysindy as ps

    library = sindy_model.feature_library
    Theta = library.fit_transform(X_observed)        # (T, n_feat)
    n_feat = Theta.shape[1]
    n_vars = len(var_names)

    # ── A2. Approximate time derivatives via finite differences ─────────────
    dt = np.diff(t).mean()
    dX_dt = np.gradient(X_observed, dt, axis=0)      # (T, n_vars)

    # Prior means from SINDy (shape: n_vars × n_feat)
    prior_means = sindy_model.coefficients()          # (n_vars, n_feat)

    # ── A3. Build PyMC model ─────────────────────────────────────────────────
    with pm.Model() as ode_model:

        # --- Priors on ODE coefficients ---
        # θ[i, j] = coefficient of feature j in the ODE for variable i
        theta = pm.Normal(
            "theta",
            mu=prior_means,                           # SINDy estimate as prior mean
            sigma=1.0,                                # adjust per domain knowledge
            shape=(n_vars, n_feat),
        )

        # --- Observation noise ---
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=2.0, shape=(n_vars,))

        # --- ODE RHS prediction via collocation ---
        # Theta_pt: (T, n_feat) as a PyTensor constant
        Theta_pt = pt.constant(Theta, dtype="float64")

        # Predicted derivative: (T, n_vars) = Theta @ theta.T
        dX_pred = pt.dot(Theta_pt, theta.T)           # (T, n_vars)

        # --- Likelihood ---
        obs = pm.Normal(
            "obs",
            mu=dX_pred,
            sigma=sigma_obs,
            observed=dX_dt,
        )

    # ── A4. Sample the posterior ─────────────────────────────────────────────
    print("\n  [Bayesian Blueprint] Starting NUTS sampler…")
    print(f"  Chains: 2  |  Samples: {n_samples}  |  Tuning: {n_tune}")
    print("  (This may take several minutes for large T or n_feat.)\n")

    with ode_model:
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            target_accept=target_accept,
            progressbar=True,
            return_inferencedata=True,
            nuts_sampler="numpyro",      # numpyro backend is ~10× faster than JAX-free
        )

    # ── A5. Quick summary ────────────────────────────────────────────────────
    try:
        import arviz as az
        print("\n  Posterior summary (theta coefficients):")
        print(az.summary(trace, var_names=["theta"]).head(20))
    except ImportError:
        print("  (Install arviz for richer posterior diagnostics: pip install arviz)")

    print("\n  [Bayesian Blueprint] Done. Use trace.posterior['theta'] for downstream analysis.")
    return trace


# ─────────────────────────────────────────────────────────────────────────────
# B.  HYBRID NEURAL ODE  (SINDy physics prior + residual MLP)
# ─────────────────────────────────────────────────────────────────────────────

def build_hybrid_neural_ode(
    sindy_model,
    var_names: list[str],
    hidden_dim: int = 32,
    poly_degree: int = 1,
    freeze_physics: bool = False,
):
    """
    Hybrid Neural ODE: SINDy physics layer + residual neural network.

    Motivation
    ──────────
    SINDy discovers sparse, interpretable equations, but real biological
    systems harbour dynamics that are:
      • Nonlinear beyond the chosen polynomial degree
      • Driven by unmeasured variables (latent confounders, gut microbiome,
        stress hormones, etc.)
      • Corrupted by structured measurement noise

    A Hybrid Neural ODE combines two components in the ODE's RHS:

        ẋ = f_physics(x; θ_SINDy)   ← interpretable, sparse, causally constrained
              + f_residual(x; φ_NN)  ← flexible, learns the unexplained remainder

    This separates what we *know* (physics prior) from what we *don't*
    (neural residual), and allows us to quantify how much the data deviates
    from the mechanistic model.

    Architecture
    ────────────
    • PhysicsLayer: evaluates the SINDy polynomial feature library and multiplies
      by learned (or frozen) coefficient matrix.  Can be frozen (θ_SINDy fixed)
      or fine-tuned (θ_SINDy trainable).
    • ResidualMLP:  small fully-connected network (2 hidden layers, ``hidden_dim``
      units, tanh activations) that outputs a correction term of the same shape
      as ẋ.
    • HybridODEFunc: wraps both into ``forward(t, x)`` for torchdiffeq.
    • NeuralODE wrapper: integrates HybridODEFunc with the adjoint method
      (O(1) memory in depth, crucial for long trajectories).

    Training recipe (recommended)
    ────────────────────────────
    1. Pre-train phase (epochs 1–50):
         Freeze PhysicsLayer (freeze_physics=True).
         Train only ResidualMLP to fit the SINDy residuals.
         Loss: MSE on integrated trajectory vs. observations.

    2. Fine-tune phase (epochs 50–200):
         Unfreeze PhysicsLayer.
         Train both components jointly with a sparsity regulariser
         (L1 on ResidualMLP parameters) to keep the residual small and
         push signal back into the interpretable physics layer.

    Parameters
    ──────────
    sindy_model  : ps.SINDy
        Fitted SINDy model used to initialise the PhysicsLayer weights.
    var_names    : list[str]
        Variable names (length = n_vars).
    hidden_dim   : int
        Width of the residual MLP's hidden layers.
    poly_degree  : int
        Polynomial degree for the physics library (must match sindy_model).
    freeze_physics : bool
        If True, the PhysicsLayer parameters are not updated during training.

    Returns
    ───────
    hybrid_model : HybridNeuralODE
        Untrained model ready for PyTorch training loop.
    loss_fn, optimizer : callable, torch.optim.Optimizer
        Standard MSE loss and Adam optimizer for the training loop.

    Usage example
    ─────────────
    model = build_hybrid_neural_ode(sindy_model, var_names)
    # Integrate:
    from torchdiffeq import odeint_adjoint as odeint
    x0_torch = torch.tensor(X[0], dtype=torch.float32)
    t_torch   = torch.tensor(t,   dtype=torch.float32)
    X_pred    = odeint(model.ode_func, x0_torch, t_torch, method='rk4')
    loss      = loss_fn(X_pred, X_torch)
    loss.backward()
    optimizer.step()
    """
    import numpy as np

    try:
        import torch
        import torch.nn as nn
        from torchdiffeq import odeint_adjoint as odeint
    except ImportError:
        raise ImportError(
            "Neural ODE blueprint requires PyTorch and torchdiffeq:\n"
            "  pip install torch torchdiffeq"
        )

    import pysindy as ps

    n_vars = len(var_names)

    # ── B1. Compute feature library dimension ────────────────────────────────
    # Fit library on a dummy 2-row array to get n_feat
    dummy = np.ones((2, n_vars))
    lib   = ps.PolynomialLibrary(degree=poly_degree, include_bias=True)
    n_feat = lib.fit_transform(dummy).shape[1]

    # Extract SINDy coefficient matrix (n_vars, n_feat) as init for physics layer
    sindy_coef = sindy_model.coefficients()            # (n_vars, n_feat)

    # ── B2. Define model components ───────────────────────────────────────────

    class PhysicsLayer(nn.Module):
        """Evaluates the SINDy polynomial RHS with learnable (or frozen) coefficients.

        Internally replicates the monomial expansion from PySINDy using
        torch operations so gradients flow through the physics parameters.
        """

        def __init__(self):
            super().__init__()
            self.coef = nn.Parameter(
                torch.tensor(sindy_coef, dtype=torch.float32),
                requires_grad=not freeze_physics,
            )
            self._poly_degree = poly_degree
            self._n_vars = n_vars

        def _feature_library(self, x: "torch.Tensor") -> "torch.Tensor":
            """Build polynomial feature vector from state x. Shape: (n_feat,)."""
            import itertools
            terms = [torch.ones(1, dtype=x.dtype, device=x.device)]  # bias
            for deg in range(1, self._poly_degree + 1):
                for combo in itertools.combinations_with_replacement(range(self._n_vars), deg):
                    term = torch.ones(1, dtype=x.dtype, device=x.device)
                    for idx in combo:
                        term = term * x[idx:idx+1]
                    terms.append(term)
            return torch.cat(terms, dim=0)  # (n_feat,)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (n_vars,)
            theta = self._feature_library(x)           # (n_feat,)
            return self.coef @ theta                   # (n_vars,)

    class ResidualMLP(nn.Module):
        """Small MLP that learns the unmodelled residual dynamics."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_vars, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, n_vars),
            )
            # Initialise output layer near zero so residual starts small
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class HybridODEFunc(nn.Module):
        """Combines physics and residual into a single ODE RHS callable."""

        def __init__(self):
            super().__init__()
            self.physics  = PhysicsLayer()
            self.residual = ResidualMLP()

        def forward(self, t: "torch.Tensor", x: "torch.Tensor") -> "torch.Tensor":
            # torchdiffeq passes (t_scalar, x_batch)
            # x may be (n_vars,) or (batch, n_vars); handle both.
            if x.dim() == 1:
                return self.physics(x) + self.residual(x)
            return torch.stack([
                self.physics(xi) + self.residual(xi) for xi in x
            ])

    class HybridNeuralODE(nn.Module):
        """Top-level model with a convenient .integrate() method."""

        def __init__(self):
            super().__init__()
            self.ode_func = HybridODEFunc()

        def integrate(
            self,
            x0: "torch.Tensor",
            t: "torch.Tensor",
            method: str = "rk4",
        ) -> "torch.Tensor":
            """
            Integrate the hybrid ODE from initial state x0 over time grid t.

            Uses the adjoint method (``odeint_adjoint``) for O(1) memory
            consumption during backpropagation — essential for 90+ time-step
            trajectories.

            Returns
            ───────
            X_pred : torch.Tensor, shape (T, n_vars)
            """
            return odeint(self.ode_func, x0, t, method=method)  # (T, n_vars)

    # ── B3. Instantiate ───────────────────────────────────────────────────────
    hybrid_model = HybridNeuralODE()
    loss_fn      = torch.nn.MSELoss()
    optimizer    = torch.optim.Adam(hybrid_model.parameters(), lr=1e-3)

    n_params_physics  = sum(p.numel() for p in hybrid_model.ode_func.physics.parameters())
    n_params_residual = sum(p.numel() for p in hybrid_model.ode_func.residual.parameters())
    print(f"\n  [Neural ODE Blueprint] Model built successfully.")
    print(f"    Physics layer params  : {n_params_physics}")
    print(f"    Residual MLP params   : {n_params_residual}")
    print(f"    Total trainable params: {n_params_physics + n_params_residual}")
    print(f"    Physics layer frozen  : {freeze_physics}")
    print("\n  To train, use the following loop (pseudocode):")
    print("""
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        x0_t = torch.tensor(X[0], dtype=torch.float32)
        t_t  = torch.tensor(t,    dtype=torch.float32)
        X_t  = torch.tensor(X,    dtype=torch.float32)
        X_pred = hybrid_model.integrate(x0_t, t_t)
        loss   = loss_fn(X_pred, X_t)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}")
    """)

    return hybrid_model, loss_fn, optimizer


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO  (python sindy_advanced_blueprints.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    print("=" * 65)
    print("  SOMA — Phase 3: Advanced Blueprints")
    print("=" * 65)
    print(
        "\nThis file contains two advanced scaffolds that extend the SINDy pipeline.\n"
        "They are designed to be read, understood, and adapted — not run as-is\n"
        "without first completing Phase 1 (sindy_stage3.py).\n"
    )

    # ── Attempt to import and build a SINDy model for demos ─────────────────
    try:
        from sindy_stage3 import _run_smoke_test
        print("Phase 1 SINDy model found. Building Hybrid Neural ODE scaffold…\n")
        model, mask, _ = _run_smoke_test()
        var_names = ["Sleep", "Mood", "Activity", "RHR", "HRV", "VO2_Max", "Glucose"]

        # Blueprint B: Neural ODE (no GPU required for scaffold)
        print("\n" + "─" * 65)
        print("  Blueprint B: Hybrid Neural ODE")
        print("─" * 65)
        try:
            hybrid, loss_fn, opt = build_hybrid_neural_ode(
                sindy_model=model,
                var_names=var_names,
                hidden_dim=32,
                freeze_physics=True,
            )
        except ImportError as e:
            print(f"  [SKIP] {e}")

        # Blueprint A: Bayesian (print description; skip sampling in demo)
        print("\n" + "─" * 65)
        print("  Blueprint A: Bayesian ODE Estimation (PyMC)")
        print("─" * 65)
        print(
            "  Requires:  pip install pymc pytensor\n"
            "  To run:    call bayesian_ode_estimation(model, X, t, var_names)\n"
            "  Note:      Sampling time scales ~ O(n_samples × n_feat × T).\n"
            "             Recommend starting with n_samples=200 on a subset of t."
        )

    except ImportError:
        print("  sindy_stage3.py not found in PYTHONPATH.")
        print("  Run `python sindy_stage3.py` first, then re-run this script.")

    print("\n" + "=" * 65)
    print("  Blueprint demo complete.  ✓")
    print("=" * 65)
