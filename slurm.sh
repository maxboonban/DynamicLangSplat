#!/bin/bash
#SBATCH --job-name=gaufre_preprocess
#SBATCH --output=gaufre_preprocess.out
#SBATCH --error=gaufre_preprocess.err
#SBATCH --time=24:00:00               # Increased time for preprocessing steps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                     # Increased memory for processing
#SBATCH --cpus-per-task=8            # Increased CPU cores for parallel processing

# Load modules
module load cuda/12.1
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Activate environment
conda activate lsplat

# Set absolute paths
WORKSPACE_DIR="/oscar/scratch/wboonban/DynamicLangSplat"
DATASET_PATH="${WORKSPACE_DIR}/data/plate"

# # Verify directory structure exists
# if [ ! -d "${DATASET_PATH}/plate/rgb/2x" ]; then
#     echo "Error: Expected directory structure not found at ${DATASET_PATH}/plate/rgb/2x"
#     exit 1
# fi

# # Change to workspace directory
# cd "${WORKSPACE_DIR}"

# # Step 1: Extract features
# echo "Running extract_features.py..."
# python extract_features.py -s "${DATASET_PATH}" -r 1

# # Step 2: Run autoencoder training and testing
# echo "Running autoencoder training..."
# cd autoencoder
# python train.py --dataset_path "${DATASET_PATH}" -r 1
# echo "Running autoencoder testing..."
# python test.py --dataset_path "${DATASET_PATH}" -r 1
# cd ..

# # Step 3: Run DINO PCA
# echo "Running DINO PCA..."
# python dino_pca.py -s "${DATASET_PATH}" -r 1

# echo "Preprocessing completed!"

# # To save the visualization instead of displaying
python visualize_dino.py --npy_path data/plate/dino/1x/000001.npy --output_path visualization.png