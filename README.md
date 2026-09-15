# EZHAL
EZHAL is an AI-assisted prototype for dynamically allocating renewable inverter support/headroom in an interconnected grid while minimizing unnecessary renewable curtailment.

# What the prototype does

1- Simulates operating states and line contingencies on a standard 9-bus test system.

2- Uses physics-constrained optimization to generate target actions.

3- Trains supervised Extra Trees models:

4- Classifier: selects the required action type.

5- Regressor: predicts where and how much support/headroom is required.

6- Applies ML decisions back to a pandapower AC power-flow model. Uses a physics-based safety layer when a direct ML action does not satisfy the security criterion.

7- Presents live ML inference and live physics validation through a Streamlit dashboard.

# Prototype results

875 simulated scenarios

659 initially insecure scenarios

Classifier Macro F1: 95.23%

82/82 selected unseen feasible test scenarios satisfied the 0.95–1.05 pu voltage criterion after AI + physics-based safety correction

Dynamic average curtailment: 1.6321 MW

5% fixed-headroom study baseline: 9.5061 MW

82.83% lower average curtailment versus that fixed 5% study baseline

# Important scope

This is a hackathon research prototype using steady-state/quasi-static AC power-flow contingency analysis. It is not a complete EMT/transient grid-forming controller and is not presented as a deployed utility system.

The fixed 5% headroom case is a study baseline, not a claim about Saudi utility operating practice.

# Run the dashboard

pip install -r requirements.txt
streamlit run app.py

The dashboard expects the trained model files and CSV result files referenced by app.py to be in the project directory.

# Computational note

The full simulation, optimization, and safety-correction stages are intentionally expensive and do not need to be rerun for every dashboard demonstration. Saved trained models and generated result files are used for the live demo.

# Core pipeline

# Simulation → Optimization → ML Training → Physics Validation → Safety Correction → Dashboard
