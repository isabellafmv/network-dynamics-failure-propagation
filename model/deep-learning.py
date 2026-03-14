import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# Layer 3 Library: Vector Autoregression
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error

# Layer 2 Library: PC Algorithm
# (Requires: pip install causal-learn)
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils


# STAGE 0: COMPLEX DATA GENERATION
np.random.seed(42)
continents = ['North America', 'Europe', 'Asia', 'South America', 'Africa']
samples_per_continent = 300 

all_data = []

for region in continents:
    # 1. Define Regional Biases
    bias_sleep = np.random.randint(-5, 5)
    bias_vo2 = np.random.randint(-5, 5)
    bias_glucose = np.random.randint(-5, 10)
    
    # Initialize Matrix (N days x 7 variables)
    data = np.zeros((samples_per_continent, 7))
    
    # Initial Baseline Values
    # Sleep(75), Mood(7), Act(5000), RHR(60), HRV(50), VO2(45), Gluc(90)
    data[0, :] = [75, 7, 5000, 60, 50, 45, 90]
    
    for t in range(1, samples_per_continent):
        # Fetch Yesterday's values
        p_sleep, p_mood, p_act, p_rhr, p_hrv, p_vo2, p_gluc = data[t-1, :]
        
        # --- THE BIOLOGICAL ENGINE ---
        
        # 1. SLEEP: Autoregressive + random variation
        new_sleep = 0.3 * p_sleep + 50 + np.random.normal(0, 8) + bias_sleep
        new_sleep = np.clip(new_sleep, 0, 100)
        
        # 2. HRV: Driven by SLEEP and MOOD
        new_hrv = 0.4 * p_hrv + 0.3 * (p_sleep - 50) + 2 * p_mood + np.random.normal(0, 5)
        
        # 3. RESTING HR (RHR): Inverse to HRV and Sleep
        new_rhr = 0.6 * p_rhr - 0.1 * (new_hrv - 50) - 0.1 * (p_sleep - 70) + np.random.normal(0, 2)
        
        # 4. MOOD: Driven by SLEEP and HRV
        new_mood = 0.4 * p_mood + 0.05 * p_sleep + 0.02 * p_hrv + np.random.normal(0, 1)
        new_mood = np.clip(new_mood, 1, 10)
        
        # 5. ACTIVITY: Driven by MOOD and VO2 Max
        new_act = 0.5 * p_act + 500 * p_mood + 50 * p_vo2 + np.random.normal(0, 1000)
        
        # 6. VO2 MAX: Slow moving, driven by ACTIVITY history
        new_vo2 = 0.95 * p_vo2 + 0.0001 * p_act + np.random.normal(0, 0.2) + (bias_vo2 * 0.01)
        
        # 7. FASTING GLUCOSE: Inverse to ACTIVITY and SLEEP
        new_gluc = 0.7 * p_gluc - 0.001 * p_act - 0.1 * (p_sleep - 70) + 30 + np.random.normal(0, 3) + bias_glucose

        data[t, :] = [new_sleep, new_mood, new_act, new_rhr, new_hrv, new_vo2, new_gluc]

    df_region = pd.DataFrame(data, columns=['Sleep', 'Mood', 'Activity', 'RHR', 'HRV', 'VO2_Max', 'Glucose'])
    df_region['Continent'] = region
    all_data.append(df_region)

df_master = pd.concat(all_data).reset_index(drop=True)
print("Stage 0 Complete: Data Generated for 7 Parameters.")
print(df_master.describe().round(1).loc[['mean', 'std', 'min', 'max']])

# STAGE 1: TRAIN/TEST SPLIT
test_continent = 'Asia'
train_cols = ['Sleep', 'Mood', 'Activity', 'RHR', 'HRV', 'VO2_Max', 'Glucose']

# Training Data: Everyone EXCEPT Asia
df_train = df_master[df_master['Continent'] != test_continent][train_cols]

# Test Data: ONLY Asia
df_test = df_master[df_master['Continent'] == test_continent][train_cols]

print(f"Stage 1 Complete: Data Split.")
print(f"   Training Samples: {len(df_train)}")
print(f"   Testing Samples: {len(df_test)} (Continent: {test_continent})")


# STAGE 2: CAUSAL DISCOVERY (PC Algorithm)

# This stage draws the "Skeleton" of the graph based on conditional independence.
# It tells us WHAT connects to WHAT, before we ask "how much" or "when".

from causallearn.search.ConstraintBased.PC import pc

print("Stage 2: Running PC Algorithm...")

# 1. Prepare Data
# PC requires a simple NumPy matrix of the training data
data_matrix = df_train.to_numpy()
labels = df_train.columns

# 2. Run the Algorithm
# alpha=0.05 is the standard statistical threshold (p-value)
# indep_test='fisherz' is the standard test for continuous data (like HR, Glucose)
cg = pc(data_matrix, alpha=0.05, indep_test='fisherz', verbose=False)

# 3. Visualization
# We convert the PC output into a NetworkX graph to draw it
G_pc = nx.DiGraph()
adj_matrix = cg.G.graph

for i in range(len(labels)):
    for j in range(len(labels)):
        # Interpretation of PC Output Matrix:
        # [j, i] == 1 AND [i, j] == -1  =>  Directed Edge (i -> j)
        if adj_matrix[j, i] == 1 and adj_matrix[i, j] == -1:
            G_pc.add_edge(labels[i], labels[j])
            
        # [j, i] == -1 AND [i, j] == -1 =>  Undirected Edge (Correlation found, direction unclear)
        elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
            if i < j: # Avoid duplicates
                G_pc.add_edge(labels[i], labels[j], style='dashed')

# 4. Plotting
plt.figure(figsize=(8, 6))

# Use circular layout because we have 7 nodes (prevents a messy hairball)
pos = nx.circular_layout(G_pc) 

nx.draw(G_pc, pos, with_labels=True, node_color='lightgreen', 
        node_size=2500, font_weight='bold', font_size=10, 
        arrowsize=20, edge_color='gray')

plt.title("Stage 2: The Causal Blueprint (PC Algorithm)\n(Solid = Causal, Dashed = Correlated)")
plt.show()

print("Stage 2 Complete: Causal Topology Discovered.")

# new step 3 (sindy)
from sindy import DataAdapter, build_causal_mask, fit_sindy_with_mask, \
                  plot_sindy_results, validate_sindy_fit, extract_equations

# Wrap your training data (daily time steps = dt=1.0)
adapter = DataAdapter(df_train, dt=1.0)

# Build causal mask from the PC adjacency matrix you already have
mask = build_causal_mask(adj_matrix, list(labels), poly_degree=1)

# Fit the ODE system (replaces VAR)
sindy_model = fit_sindy_with_mask(adapter.X, adapter.t, mask, list(labels))

# Stage 4-equivalent: dual heatmap (coefficients + sparsity pattern)
plot_sindy_results(sindy_model, list(labels))

# Stage 6-equivalent: integrate ODEs and report RMSE per variable
validate_sindy_fit(sindy_model, adapter.X, adapter.t, list(labels))

