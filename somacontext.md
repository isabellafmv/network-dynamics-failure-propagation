SOMA is a comprehensive system designed to serve as a "blueprint for longevity" by allowing users to **understand, predict, and steer** their personal health. Developed by Julia, Isabella, Nina, and Zani for the **Bloom-Lab Hackathon 2025**, the platform transitions health tracking from single-purpose data collection to a multi-purpose, integrated system.

## Core Methodology

The system operates through a three-step cycle to manage physiological well-being:
1. **Quantify:** Measure the current physiological state.
2. **Forecast:** Predict future health trajectories.
3. **Recommend:** Suggest specific interventions to optimize health outcomes

### Data Integration & Pillars

SOMA aggregates data across more than **150 measurable variables**, including blood pressure (BP), heart rate variability (HRV), lipids, microbiome, sleep stages, and omics panels. This data is categorized into five primary "lenses":

- **Molecular / Omics:** Genetic and metabolic data
- **Lifestyle & Behavior:** Routine actions and habits
- **Clinical & Functional:** Medical records and physical capabilities
- **Environment & Exposures:** External factors impacting health
- **Organ System Hubs:** Mapping data to specific systems such as cardio, metabolic, respiratory, renal, and immune

## Three Lenses of Personal Health

The platform provides users with three distinct ways to view and interact with their health data:

### 1. Pillars of Lifestyle

Focuses on six foundational areas to achieve **Optimal Longevity** (benchmarked at a 56% baseline in examples):

- Nutrition.
- Physical Activity.
- Restorative Sleep.
- Stress Management.
- Social Connections.
- Avoidance of Risky Substances.

### 2. Digital Twin

A virtual representation of the user’s body used to model complex physiological interactions. SOMA builds a **complete graph** of the user's data and applies the **Peter-Clark algorithm** to infer directionality and reduce complexity. It then utilizes **Vector Autoregression (VAR)** to determine "edge weights," or the strength of relationships between different health variables.

The modeling architecture of SOMA is designed to move beyond simple correlation by using causal inference to map how different health variables actually influence one another.

The process follows a specific multi-stage pipeline:
#### 1. Building the Complete Graph

The system begins by mapping all data availability from manual inputs, connected devices (like Oura, Garmin, or Strava), and medical records. It creates a "complete graph" where every measurable variable—from sleep stages and HRV to glucose levels and mood—is initially connected.

#### 2. Causal Discovery (Peter-Clark Algorithm)

To transform a mess of correlations into a functional map, SOMA applies the **Peter-Clark (PC) algorithm**.

- **Directionality**: It reduces the complete graph by identifying which variables have a "cause-and-effect" relationship rather than just moving together by chance.
- **Refinement**: This results in a Directed Acyclic Graph (DAG) that acts as the "Causal Blueprint" for the user’s body.
#### 3. Quantifying Relationships (VAR & Edge Weights)

Once the direction of the relationship is known (e.g., how Activity affects Sleep), the system uses **Vector Autoregression (VAR)**.

- **Edge Weights**: This step calculates the "weight" or strength of these connections.
- **Personalization**: It determines exactly how much a 10% increase in deep sleep might improve your HRV or metabolic markers specifically for _your_ unique physiology.
#### 4. Simulation and Intervention

Finally, interventions are "injected" into the mathematical system. This allows the model to run "what-if" scenarios:
- **Baseline vs. Intervention**: The model forecasts a future trajectory without changes versus a trajectory where specific lifestyle "pillars" are optimized
- **Threshold Detection**: It identifies "Acute intervention thresholds," which are specific points in time where the model predicts a significant physiological decline unless a recommended action is taken
### 3. Forecast

Visualizes the user's health trajectory. It compares a **Future outlook without Intervention** against a **Future outlook with proposed Interventions**, identifying "Acute intervention thresholds" where action is necessary to prevent decline.

## Technical Foundations & Modeling

SOMA is trained on massive global datasets to ensure accuracy across diverse demographics:

- **HALL (Human Aging and Longevity Landscape):** >250,000 samples (ages 1–119).
    
- **WHO Mortality Database:** >5,000,000 samples (ages 0–116).
    
- **Human Mortality Database:** >10,000,000 samples from 40 developed countries.
    
- **UK Biobank:** >500,000 adults (ages 40–69) focusing on genetics and lifestyle.
    

### Validation Methodology

Using three distinct types of data to ensure the model wasn't just "hallucinating" patterns:

- **Linear Dependencies**: Using dummy medical time-series data to confirm the PC algorithm correctly identifies known links.
    
- **Random Data**: Testing the model against completely random data to ensure it correctly identifies a lack of causal structure (avoiding false positives).
    
- **Real-World Generalization**: Testing the model on weather time-series data and unseen continental datasets (e.g., Asia) to prove that the underlying logic holds true across different environments
