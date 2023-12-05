#!/bin/bash -l

echo "Executing main.sh..."

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
    job_name="./scripts/log_files/llms_nodes_$num_nodes"
    output_filename="${job_name}.out"
    error_filename="${job_name}.err"
    echo "Current path $PWD"
    echo "Trying to call ./scripts/main.job"
    sbatch --nodes=$num_nodes --job-name=$job_name --output=$output_filename --error=$error_filename ./scripts/main.job;
    #./scripts/test.sh
else
    python ./src/main.py
fi
