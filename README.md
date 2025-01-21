# Automated Experiment Runner for learn_as_you_go framework

This repository contains python and bash scripts designed to automate the process of running Python experiments. The script executes a series of **pre-scripts**, followed by the main experimental scripts multiple times, and concludes with **post-scripts** for analysis or cleanup.

---

## Script Workflow

1. **Pre-Scripts**: 
   - Scripts required to prepare the experimental setup (e.g., variable initialization, image tiling, ground truth creation, agent configuration, environment configuration).
   - These are executed one after the other.

2. **Main Scripts**: 
   - The core experimental Python scripts are executed concurrently for a configurable number of runs (set to 3 currently).

3. **Post-Scripts**: 
   - Analysis or final cleanup scripts that are executed sequentially after the experiments.

---

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/gowrysailajav/learn_as_you_go.git
2. Run full_automation.py
   ```bash
   pyton full_automation.py
   
