#!/bin/bash -l

# Check which computer/cluster to execute the tests on
if [[ $PWD == *"scratch"* ]]; then
    on_cluster=1
else
    on_cluster=0
fi

# Number of nodes
num_nodes=2

cd "../" || exit
if [[ $on_cluster == 1 ]]; then
    job_name="./logs/llms_nodes_$num_nodes"
    output_filename="${job_name}.out"
    error_filename="${job_name}.err"
    sbatch --nodes=$num_nodes --job-name=$job_name --output=$output_filename -error=$error_filename ./src/main.py
else
    python ./src/main.py
fi