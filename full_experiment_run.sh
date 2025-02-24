#!/bin/bash

# Number of times to run each experiment
NUM_RUNS=3

# Define the list of Python scripts with their arguments
PYTHON_SCRIPTS=(
    "Testing/c1_ps_sum.py"
)

# Python scripts to run before the main scripts
POST_SCRIPTS=(
    # "Analysis/plot1.py"
    "Analysis/plot2.py"
)

# Python scripts to run after the main scripts
PRE_SCRIPTS=(
    "Training/s0_variables.py"
    "Testing/config_file.py"
    "Testing/s0.0_image_tiling.py"
    "Testing/s0.1_image_color_coding.py"
    "Testing/s0.2.2_population_without_elevation_testig.py"
    "Testing/s1.2_grid_conf_training.py"
    "Testing/s4.2_cell_conf_training.py"
    "Testing/s2_1_agent_conf.py"
    "Testing/s5.1_aoi_generation.py"
    "Testing/s3_env_conf.py"
    "Testing/counter_update.py"
)

# Function to run a list of scripts sequentially
run_scripts_sequentially() {
    local scripts=("$@")
    for script in "${scripts[@]}"
    do
        echo "Running script: $script"
        eval "python $script" &
        sleep 100
    done
}

# Run pre-scripts one after the other
echo "Running pre-scripts..."
run_scripts_sequentially "${PRE_SCRIPTS[@]}"

# Loop through each run
for ((i=1; i<=NUM_RUNS; i++))
do
    echo "Starting Exp $i..."

    # Loop through each main Python script
    for SCRIPT in "${PYTHON_SCRIPTS[@]}"
    do
        echo "Running script: $SCRIPT"
        # Extract the script name for logging
        SCRIPT_NAME=$(echo $SCRIPT | awk '{print $1}')
        # Run the current Python script in the background and redirect output to log files
        eval "python $SCRIPT" &
    done

    # Wait for all background processes to finish
    wait
    echo "Exp $i main scripts completed."
done

# Run post-scripts one after the other
echo "Running post-scripts..."
run_scripts_sequentially "${POST_SCRIPTS[@]}"

echo "All experiments completed."
