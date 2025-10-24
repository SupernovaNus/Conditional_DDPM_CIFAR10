#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH -C gpu
#SBATCH --account=m4287
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J cifar-ddp
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=pchang3@lbl.gov

module load conda
conda activate fair

# for DDP
export MASTER_ADDR=$(hostname)      ##
cmd="python3 train_ddp.py"

set -x
srun -l \
    bash -c "
    source export_DDP_vars.sh       ##
    $cmd
    " 