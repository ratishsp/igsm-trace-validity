#!/bin/bash
#SBATCH --job-name=igsm_eval
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=8 --time=06:00:00
# Usage: sbatch slurm/eval_template.sh <checkpoint> <output_dir> [evaluate.py flags]
# IGSM_ROOT: the iGSM generator checkout (default ./iGSM). Add --account and --partition for your site.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$2"
srun python -u evaluate.py --checkpoint "$1" --output_dir "$2" --difficulty med \
     --igsm_root "${IGSM_ROOT:-./iGSM}" --gen_batch 32 "${@:3}"
