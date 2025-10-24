#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH -C gpu
#SBATCH --account=m4287
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J cifar-single-gpu
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=pchang3@lbl.gov

module load conda
conda activate fair

cmd="python3 train_single_gpu.py"

set -x
srun -l \
    bash -c "
    $cmd
    " 