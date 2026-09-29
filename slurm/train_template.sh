#!/bin/bash
#SBATCH --job-name=igsm_train
#SBATCH --nodes=4 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=7 --time=42:00:00
# 16 GPUs x micro-batch 32 = batch 512, the setting of every run in the paper. One task per GPU (no torchrun).
# Usage: sbatch slurm/train_template.sh <output_dir> [igsm_train.py flags, e.g. --swap_traces]
# IGSM_ROOT: the iGSM generator checkout (default ./iGSM). Add --account and --partition for your site.
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))   # a fixed port collides when two jobs share a node
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$1"
srun python -u igsm_train.py --mode train --igsm_root "${IGSM_ROOT:-./iGSM}" --output_dir "$1" \
     --grad_accum_steps 1 --sampling light --seed 42 --eval_steps 2000 --eval_problems 128 "${@:2}"
