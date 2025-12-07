import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Layer 2 Library: PC Algorithm (pip install causal-learn)
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

# Layer 3 Library: Vector Autoregression
from statsmodels.tsa.api import VAR


# STAGE 0: LOAD REAL DATA & RESHAPE TO LONG FORMAT

# Adjust path if needed
df_raw = pd.read_excel("/Users/isabellamueller-vogt/Library/Mobile Documents/com~apple~CloudDocs/08 - side quests/network-dynamics-failure-propagation/model/data/cohort_study_5000.xlsx")

print("Stage 0: Data loaded.")
print(f"Raw shape: {df_raw.shape}")
print("Columns (first 20):", df_raw.columns[:20].tolist())

# The dataset has weekly measures for 1 year (52 weeks) on 5000 participants.
# Columns look like: vo2_w1, rhr_w1, ..., circ_w52
# We'll reshape to long format: one row = (participant, week) with 6 biomarker vars.

biomarkers = ['vo2', 'rhr', 'sys', 'dia', 'bfperc', 'circ']

long = pd.wide_to_long(
    df_raw,
    stubnames=biomarkers,
    i='participant_id',
    j='week',
    sep='_w',
    suffix='\\d+'
).reset_index()

print("\nLong-format data:")
print(long.head())
print("Long shape:", long.shape)

# ============================================================
# STAGE 1: RANDOM TRAIN/TEST SPLIT (BY PARTICIPANT)
# ============================================================

test_size = 0.2
participants = df_raw['participant_id'].unique()

train_ids, test_ids = train_test_split(
    participants,
    test_size=test_size,
    random_state=42
)

train_long = long[long['participant_id'].isin(train_ids)].copy()
test_long = long[long['participant_id'].isin(test_ids)].copy()

print("\nStage 1: Train/Test split (by participant).")
print(f"  Train participants: {len(train_ids)}")
print(f"  Test participants:  {len(test_ids)}")
print(f"  Train rows (person-weeks): {train_long.shape[0]}")
print(f"  Test rows  (person-weeks): {test_long.shape[0]}")

# For PC (causal discovery), we treat each person-week as an i.i.d. sample.
df_train_pc = train_long[biomarkers].dropna()

print("\nPC training matrix shape:", df_train_pc.shape)

# For VAR (temporal dynamics), we aggregate weekly averages across TRAIN participants
df_train_ts = (
    train_long
    .groupby('week')[biomarkers]
    .mean()
    .sort_index()
)

df_test_ts = (
    test_long
    .groupby('week')[biomarkers]
    .mean()
    .sort_index()
)

print("Time-series (TRAIN) shape:", df_train_ts.shape)  # expected ~ (52, 6)
print("Time-series (TEST) shape: ", df_test_ts.shape)


# ============================================================
# STAGE 2: CAUSAL DISCOVERY (PC Algorithm, improved)
# ============================================================

print("\nStage 2: Running PC Algorithm on person-week samples...")

data_matrix = df_train_pc.to_numpy()
labels = list(df_train_pc.columns)
n_vars = len(labels)

# Run PC
cg = pc(
    data_matrix,
    alpha=0.05,           # significance threshold
    indep_test='fisherz', # for continuous data
    verbose=False
)

# Extract adjacency matrix from CPDAG
adj = cg.G.graph

edges = []  # (u, v, edge_type)
parents = {v: set() for v in labels}

for i in range(n_vars):
    for j in range(i + 1, n_vars):  # only upper triangle (i < j)
        a_ij = adj[i, j]
        a_ji = adj[j, i]

        # No edge
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

        # undirected (i — j): ambiguous direction in CPDAG
        elif a_ij == -1 and a_ji == -1:
            u, v, etype = labels[i], labels[j], 'undirected'
            edges.append((u, v, etype))
            # Treat both as candidate parents for modeling
            parents[u].add(v)
            parents[v].add(u)

        # bidirected (rare in plain PC, included for completeness)
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

plt.figure(figsize=(8, 6))
pos = nx.circular_layout(G_pc)

nx.draw_networkx_nodes(G_pc, pos, node_color='#2F687D', node_size=2500)
nx.draw_networkx_labels(G_pc, pos, font_weight='bold', font_size=10, font_color='#FDFCF4')

solid_edges = [(u, v) for u, v, d in G_pc.edges(data=True) if d.get('style') == 'solid']
dashed_edges = [(u, v) for u, v, d in G_pc.edges(data=True) if d.get('style') == 'dashed']

nx.draw_networkx_edges(
    G_pc, pos,
    edgelist=solid_edges,
    arrows=True,
    arrowsize=25,
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
plt.show()


# ============================================================
# STAGE 3: TEMPORAL DYNAMICS (VAR Model, improved)
# ============================================================

print("\nStage 3: Fitting VAR model on weekly TRAIN averages...")

model = VAR(df_train_ts)

# Let VAR choose lag order up to 4 using AIC
results = model.fit(maxlags=4, ic='aic')
lag_order = results.k_ar
var_names = list(df_train_ts.columns)

print(f"Stage 3 Complete: VAR fitted with lag order = {lag_order}")
print("Variables:", var_names)

# Coefficient tensor: (lags, n_vars, n_vars)
coefs = results.coefs
print("Coefficient tensor shape:", coefs.shape)

# Example causal check: Does higher VO2 predict lower RHR?
print("\nChecking logic: Does higher VO2 predict lower RHR?")

try:
    vo2_idx = var_names.index('vo2')
    rhr_idx = var_names.index('rhr')
except ValueError as e:
    raise ValueError("Expected 'vo2' and 'rhr' in time-series columns") from e

vo2_to_rhr = []
for lag in range(1, lag_order + 1):
    coef_lag = coefs[lag - 1, rhr_idx, vo2_idx]  # rhr_t ← vo2_{t-lag}
    vo2_to_rhr.append(coef_lag)
    sign = "negative (higher fitness → lower RHR)" if coef_lag < 0 else "positive"
    print(f"  Lag {lag}: coef = {coef_lag:.5f} ({sign})")

total_effect = sum(vo2_to_rhr)
print(f"\nTotal VO2→RHR effect across {lag_order} lags: {total_effect:.5f} "
      f"({'overall negative' if total_effect < 0 else 'overall positive'})")

# Optional: Granger causality test
try:
    causality_res = results.test_causality(
        caused='rhr',
        causing=['vo2'],
        kind='wald'
    )
    print("\nGranger causality test: VO2 → RHR")
    print(f"  p-value: {causality_res.pvalue:.4f}")
    print("  => ",
          "Reject no-causality (VO2 helps predict RHR)"
          if causality_res.pvalue < 0.05
          else "Cannot reject no-causality at 0.05 level")
except Exception as e:
    print("Could not run Granger causality test:", e)


# ============================================================
# STAGE 4: VISUALIZE LAG-1 COEFFICIENT MATRIX
# ============================================================

print("\nStage 4: Visualizing lag-1 coefficient matrix...")

if lag_order >= 1:
    matrix_l1 = pd.DataFrame(
        coefs[0],             # lag 1 = index 0
        index=var_names,      # rows: effect at time t
        columns=var_names     # cols: cause at time t-1
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_l1, annot=True, fmt=".2f", cmap='RdBu_r', center=0)
    plt.title("Lag-1 'Health Logic' Map\nColumns: Cause (t-1) → Rows: Effect (t)")
    plt.ylabel("Effect (today)")
    plt.xlabel("Cause (yesterday)")
    plt.tight_layout()
    plt.show()
else:
    print("Model chose lag_order = 0: no temporal structure to visualize.")


# ============================================================
# STAGE 5: INTERVENTION SIMULATION (IRFs, manual plotting)
# ============================================================

print("\nStage 5: Running Intervention Simulations via IRFs...")

irf = results.irf(15)        # 15-week horizon
irf_matrix = irf.irfs        # shape: steps x n_vars x n_vars
steps = irf_matrix.shape[0]
t = np.arange(steps)
idx = {name: i for i, name in enumerate(var_names)}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scenario 1: VO2 ↑ → RHR response
imp = idx['vo2']
resp = idx['rhr']
axes[0].plot(t, irf_matrix[:, resp, imp], linewidth=2)
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].set_title("IRF: VO2 ↑ → RHR Response")
axes[0].set_xlabel("Weeks after intervention")
axes[0].set_ylabel("IRF magnitude")
axes[0].grid(alpha=0.3)

# Scenario 2: Body fat % ↑ → Systolic BP response
imp = idx['bfperc']
resp = idx['sys']
axes[1].plot(t, irf_matrix[:, resp, imp], linewidth=2, color='purple')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title("IRF: Body fat ↑ → Systolic BP Response")
axes[1].set_xlabel("Weeks after intervention")
axes[1].grid(alpha=0.3)

fig.suptitle("Stage 5: Intervention IRFs (Weekly Dynamics)", fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# STAGE 6: VALIDATION ON UNSEEN PARTICIPANTS
# ============================================================

print("\nStage 6: Validation on held-out participants (weekly means)...")

test_values = df_test_ts.values
lag_order = results.k_ar

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

# Example: validate on systolic BP ("sys")
target = 'sys'
plt.figure(figsize=(10, 4))
limit = len(pred_df)  # all weeks after initial lags
plt.plot(act_df[target].values[:limit], label='Actual (test participants)', color='black', alpha=0.6)
plt.plot(pred_df[target].values[:limit], label='Model prediction', color='red', linestyle='--')
plt.title(f"Validation: Predicting {target.upper()} on Unseen Participants")
plt.xlabel("Weeks (after initial lags)")
plt.ylabel(target)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

rmse = np.sqrt(mean_squared_error(act_df[target], pred_df[target]))
print(f"Validation Complete. RMSE on '{target}' (test weekly means): {rmse:.2f}")
