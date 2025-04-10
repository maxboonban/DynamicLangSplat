#!/bin/bash
#SBATCH --job-name=warp_run
#SBATCH --output=warp_run.out
#SBATCH --error=warp_run.err
#SBATCH --time=02:00:00               # Adjust as needed
#SBATCH --partition=gpu               # Or the appropriate GPU partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --mem=16G                     # Adjust memory as needed
#SBATCH --cpus-per-task=4            # Adjust CPU cores as needed

# Load modules
module load cuda/12.1.
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Activate environment
conda activate lsplat


# Run your Python script
