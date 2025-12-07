import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Causal-learn PC Algorithm
from causallearn.search.ConstraintBased.PC import pc

# VAR model
from statsmodels.tsa.api import VAR

# ============================================================
# STAGE 0: LOAD & PREP WEATHER DATA
# ============================================================

# This assumes ghcn_clean.csv from your previous script:
# columns: station, date, variable, value
ghcn_file = "/Users/isabellamueller-vogt/Library/Mobile Documents/com~apple~CloudDocs/08 - side quests/network-dynamics-failure-propagation/model/data/ghcn_clean_small.csv"

print("Loading weather data from:", ghcn_file)
df_long = pd.read_csv(ghcn_file, parse_dates=["date"])

# Keep a specific set of variables
weather_vars = ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]
df_long = df_long[df_long["variable"].isin(weather_vars)].copy()

# Pivot to wide format: one row per (station, date)
df_wide = (
    df_long
    .pivot_table(index=["station", "date"],
                 columns="variable",
                 values="value")
    .reset_index()
)

print("Wide dataframe shape (station-date rows):", df_wide.shape)

# Drop rows where any of the chosen vars are missing
df_wide = df_wide.dropna(subset=weather_vars)
print("After dropping NA rows:", df_wide.shape)


# ============================================================
# STAGE 1: TRAIN/TEST SPLIT (BY LOCATION, NOT TIME)
# ============================================================

print("\nStage 1: Train/Test split by LOCATION (stations)")

test_size = 0.2  # 20% of stations held out
random_state = 42

# 1. Get unique station IDs
stations = df_wide["station"].unique()
print("Total unique stations:", len(stations))

# 2. Split stations into train/test sets
from sklearn.model_selection import train_test_split

train_stations, test_stations = train_test_split(
    stations,
    test_size=test_size,
    random_state=random_state
)

print("Train stations:", len(train_stations))
print("Test stations: ", len(test_stations))

# 3. Split the FULL dataset by station
train_data = df_wide[df_wide["station"].isin(train_stations)].copy()
test_data  = df_wide[df_wide["station"].isin(test_stations)].copy()

print("Train rows (station-days):", train_data.shape[0])
print("Test rows  (station-days):", test_data.shape[0])

# ------------------------------------------------------------
# DATA FOR PC (IID SAMPLES ACROSS LOCATIONS + DAYS)
# ------------------------------------------------------------

df_train_pc = train_data[weather_vars].dropna()
print("\nPC training matrix shape (train stations, all days):", df_train_pc.shape)

# ------------------------------------------------------------
# DATA FOR VAR (AGGREGATED DAILY FIELD OVER TRAIN STATIONS)
# ------------------------------------------------------------

# Aggregate across TRAIN stations by date (spatial mean field)
df_train_ts = (
    train_data
    .groupby("date")[weather_vars]
    .mean()
    .sort_index()
)

df_test_ts = (
    test_data
    .groupby("date")[weather_vars]
    .mean()
    .sort_index()
)

print("\nTime-series (TRAIN, aggregated over locations) shape:", df_train_ts.shape)
print("Time-series (TEST, aggregated over locations) shape: ", df_test_ts.shape)

print("Train date range:", df_train_ts.index.min(), "to", df_train_ts.index.max())
print("Test  date range:", df_test_ts.index.min(), "to", df_test_ts.index.max())


# ============================================================
# STAGE 2: CAUSAL DISCOVERY (PC Algorithm)
# ============================================================

print("\nStage 2: Running PC Algorithm on daily samples...")

data_matrix = df_train_pc.to_numpy()
labels = list(df_train_pc.columns)
n_vars = len(labels)

cg = pc(
    data_matrix,
    alpha=0.05,           # significance threshold
    indep_test='fisherz', # continuous vars
    verbose=False
)

# cg.G is a CPDAG (graph structure)
adj = cg.G.graph  # adjacency matrix

edges = []  # (u, v, edge_type)
parents = {v: set() for v in labels}

for i in range(n_vars):
    for j in range(i + 1, n_vars):
        a_ij = adj[i, j]
        a_ji = adj[j, i]

        if a_ij == 0 and a_ji == 0:
            continue

        # i -> j
        if a_ji == 1 and a_ij == -1:
            u, v, etype = labels[i], labels[j], 'directed'
            edges.append((u, v, etype))
            parents[v].add(u)

        # j -> i
        elif a_ij == 1 and a_ji == -1:
            u, v, etype = labels[j], labels[i], 'directed'
            edges.append((u, v, etype))
            parents[v].add(u)

        # undirected (ambiguous in CPDAG)
        elif a_ij == -1 and a_ji == -1:
            u, v, etype = labels[i], labels[j], 'undirected'
            edges.append((u, v, etype))
            parents[u].add(v)
            parents[v].add(u)

        # bidirected (rare)
        elif a_ij == 1 and a_ji == 1:
            u, v, etype = labels[i], labels[j], 'bidirected'
            edges.append((u, v, etype))
            parents[u].add(v)
            parents[v].add(u)

print("\nStage 2 Complete: Causal topology discovered.\n")
print("Directed / candidate parents per variable:")
for v in labels:
    print(f"  {v}: {sorted(parents[v])}")

# ---- Visualization with networkx ----
G_pc = nx.DiGraph()
G_pc.add_nodes_from(labels)

for u, v, etype in edges:
    if etype == 'directed':
        G_pc.add_edge(u, v, style='solid')
    elif etype in ('undirected', 'bidirected'):
        G_pc.add_edge(u, v, style='dashed')

plt.figure(figsize=(7, 6))
pos = nx.circular_layout(G_pc)

nx.draw_networkx_nodes(G_pc, pos, node_color='lightblue', node_size=2500)
nx.draw_networkx_labels(G_pc, pos, font_weight='bold', font_size=10)

solid_edges = [(u, v) for u, v, d in G_pc.edges(data=True) if d.get('style') == 'solid']
dashed_edges = [(u, v) for u, v, d in G_pc.edges(data=True) if d.get('style') == 'dashed']

nx.draw_networkx_edges(
    G_pc, pos,
    edgelist=solid_edges,
    arrows=True,
    arrowsize=20,
    arrowstyle='-|>',
    edge_color='black',
    width=2
)
nx.draw_networkx_edges(
    G_pc, pos,
    edgelist=dashed_edges,
    arrows=False,
    style='dashed',
    edge_color='gray',
    width=2
)

plt.title("Stage 2: Causal Blueprint (PC CPDAG)\nSolid = directed, dashed = ambiguous")
plt.axis("off")
plt.tight_layout()
plt.show()

# ============================================================
# STAGE 3: TEMPORAL DYNAMICS (VAR Model)
# ============================================================

print("\nStage 3: Fitting VAR model on daily TRAIN series...")

model = VAR(df_train_ts)
results = model.fit(maxlags=7, ic='aic')  # up to 7-day lag, select by AIC
lag_order = results.k_ar
var_names = list(df_train_ts.columns)

print(f"Stage 3 Complete: VAR fitted with lag order = {lag_order}")
print("Variables:", var_names)

coefs = results.coefs  # shape: (lags, n_vars, n_vars)
print("Coefficient tensor shape:", coefs.shape)

# Example: Does higher TMAX affect SNWD (snow depth) over time?
print("\nChecking logic: Does higher TMAX predict changes in SNWD?")

try:
    tmax_idx = var_names.index("TMAX")
    snwd_idx = var_names.index("SNWD")
except ValueError as e:
    raise ValueError("Expected 'TMAX' and 'SNWD' in time-series columns") from e

tmax_to_snwd = []
for lag in range(1, lag_order + 1):
    coef_lag = coefs[lag - 1, snwd_idx, tmax_idx]  # SNWD_t ← TMAX_{t-lag}
    tmax_to_snwd.append(coef_lag)
    sign = "negative (warmer → less snow depth)" if coef_lag < 0 else "positive (warmer → more snow depth)"
    print(f"  Lag {lag}: coef = {coef_lag:.5f} ({sign})")

total_effect = sum(tmax_to_snwd)
print(f"\nTotal TMAX→SNWD effect across {lag_order} lags: {total_effect:.5f} "
      f"({'overall negative' if total_effect < 0 else 'overall positive'})")

# ============================================================
# STAGE 4: VISUALIZE LAG-1 COEFFICIENT MATRIX
# ============================================================

print("\nStage 4: Visualizing lag-1 coefficient matrix...")

if lag_order >= 1:
    matrix_l1 = pd.DataFrame(
        coefs[0],
        index=var_names,   # rows: effect at time t
        columns=var_names  # cols: cause at time t-1
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix_l1, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Lag-1 Weather Effect Map\nColumns: Cause (t-1) → Rows: Effect (t)")
    plt.ylabel("Effect (today)")
    plt.xlabel("Cause (yesterday)")
    plt.tight_layout()
    plt.show()
else:
    print("Model chose lag_order = 0: no temporal structure to visualize.")

# ============================================================
# STAGE 5: INTERVENTION SIMULATION (IRFs)
# ============================================================

print("\nStage 5: Running Intervention Simulations via IRFs...")

irf = results.irf(15)  # 15-day horizon
irf_matrix = irf.irfs  # shape: steps x n_vars x n_vars
steps = irf_matrix.shape[0]
t = np.arange(steps)
idx = {name: i for i, name in enumerate(var_names)}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scenario 1: TMAX ↑ → SNWD response
imp = idx["TMAX"]
resp = idx["SNWD"]
axes[0].plot(t, irf_matrix[:, resp, imp], linewidth=2)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("IRF: TMAX ↑ → SNWD Response")
axes[0].set_xlabel("Days after intervention")
axes[0].set_ylabel("IRF magnitude")
axes[0].grid(alpha=0.3)

# Scenario 2: PRCP ↑ → SNWD response
imp = idx["PRCP"]
resp = idx["SNWD"]
axes[1].plot(t, irf_matrix[:, resp, imp], linewidth=2, color="purple")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("IRF: PRCP ↑ → SNWD Response")
axes[1].set_xlabel("Days after intervention")
axes[1].grid(alpha=0.3)

fig.suptitle("Stage 5: Intervention IRFs (Daily Weather Dynamics)", fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# STAGE 6: FORECASTING & VALIDATION ON HELD-OUT PERIOD
# ============================================================

print("\nStage 6: Validation on held-out test period (aggregated over locations)...")

def validate_var_on_heldout(df_test_ts, results, targets=None):
    """
    Validate a fitted VAR model on a held-out time period (df_test_ts).

    df_test_ts : DataFrame
        Test time series with datetime index and columns matching the VAR model.
    results    : VARResults
        Fitted VAR model.
    targets    : list[str] or None
        Which variables to validate/plot. If None, validate all columns.
    """
    var_names = list(df_test_ts.columns)
    if targets is None:
        targets = var_names
    else:
        # only keep targets that actually exist in the data
        targets = [t for t in targets if t in var_names]

    if not targets:
        print("No valid target variables found in df_test_ts; nothing to validate.")
        return

    test_values = df_test_ts.values
    lag_order = results.k_ar

    if len(test_values) <= lag_order:
        print("Not enough test points after lag_order to perform validation.")
        return

    # Build rolling forecasts over the test period
    predictions = []
    actuals = []

    for t_step in range(lag_order, len(test_values)):
        # last `lag_order` observations from TEST group
        hist = test_values[t_step - lag_order:t_step]
        pred = results.forecast(y=hist, steps=1)
        predictions.append(pred[0])
        actuals.append(test_values[t_step])

    pred_df = pd.DataFrame(predictions, columns=var_names)
    act_df = pd.DataFrame(actuals, columns=var_names)

    # Compute and plot for each target variable
    for target in targets:
        plt.figure(figsize=(10, 4))
        limit = len(pred_df)
        plt.plot(
            act_df[target].values[:limit],
            label="Actual (held-out mean field)",
            color="black",
            alpha=0.6,
        )
        plt.plot(
            pred_df[target].values[:limit],
            label="Model prediction",
            color="red",
            linestyle="--",
        )
        plt.title(f"Validation: Predicting {target} on Held-Out Days\n"
                  f"(aggregated over held-out stations)")
        plt.xlabel("Days (after initial lags)")
        plt.ylabel(target)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        rmse = np.sqrt(mean_squared_error(act_df[target], pred_df[target]))
        print(f"Validation RMSE on '{target}' (test days): {rmse:.3f}")


# ---- Call the function for multiple weather variables ----
targets_to_check = ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]
validate_var_on_heldout(df_test_ts, results, targets=targets_to_check)
