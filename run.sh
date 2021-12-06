#!/bin/bash
#SBATCH --tasks-per-node=2  # set to number of GPUs per node
#SBATCH --cpus-per-task=20  # set to number of cpus per node / number of tasks per node

# load modules if needed

export TOTAL_GPUS=${SLURM_NTASKS}  # num nodes * num gpus per node
export GPUS_PER_NODE=2
export LOG_DIR="/path/to/logs"

LOG_FILE=${LOG_DIR}/${TOTAL_GPUS}.log
ERR_LOG=${LOG_DIR}/${TOTAL_GPUS}.err
CONFIG=${LOG_DIR}/config.json

# srun or mpirun depending on your system
srun python train.py \
    --task=${TASK} \
    --batch_size=${BATCH_SIZE} \
    --num_epochs=${NUM_EPOCHS} \
    --num_nodes=${SLURM_NNODES} \
    --log_dir=${LOG_DIR} 2>${ERR_LOG} 1>${LOG_FILE}