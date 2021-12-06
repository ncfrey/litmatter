#!/bin/bash

# Call with "./run.sh <batch size> <epochs> <num nodes>"
export BATCH_SIZE=$1
export NUM_EPOCHS=$2
export NUM_NODES=$3

export LOG_DIR="/path/to/logs"
mkdir -p ${LOG_DIR}

sbatch --output=${LOG_DIR}"/%j.log" \
    --gres=gpu:volta:2 \
    --nodes ${NUM_NODES} \
    run.sh ${BATCH_SIZE} \
    ${NUM_EPOCHS}