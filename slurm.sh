#!/bin/bash
#SBATCH --job-name=gaufre_preprocess
#SBATCH --output=slurm/gaufre_preprocess1.out
#SBATCH --error=slurm/gaufre_preprocess1.err
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

# python print_npy_files.py input_data/plate/clip/1x/embeddings.npy

# Set dataset name and resolution
DATASET_NAME="plate"
RESOLUTION=1

# Set absolute paths
WORKSPACE_DIR="/oscar/scratch/wboonban/DynamicLangSplat"
cd "${WORKSPACE_DIR}"

# Set dataset path
DATASET_PATH="/oscar/scratch/wboonban/DynamicLangSplat/input_data/${DATASET_NAME}"
OUTPUT_PATH="/oscar/scratch/wboonban/DynamicLangSplat/output/${DATASET_NAME}"

# # Step 0: Run data preprocessing
# echo "Running data preprocessing..."
# sh data_preprocess.sh "${DATASET_NAME}" "${RESOLUTION}"

# Step 1: Extract features
echo "Running extract_features.py..."
python extract_features.py -s "${DATASET_PATH}" -r "${RESOLUTION}"

# Step 2: Run autoencoder training and testing
echo "Running autoencoder training..."
cd autoencoder
python train.py --dataset_path "${DATASET_PATH}" -r "${RESOLUTION}"
echo "Running autoencoder testing..."
python test.py --dataset_path "${DATASET_PATH}" -r "${RESOLUTION}"
cd ..

# # Step 3: Run DINO PCA
# echo "Running DINO PCA..."
# python dino_pca.py -s "${DATASET_PATH}" -r "${RESOLUTION}"

# # Step 4: Train and render GauFRe
# echo "Training and rendering GauFRe model..."
# # For real-world scenes
# bash scripts/trainval_real.sh "${DATASET_PATH}" "${OUTPUT_PATH}"

# echo "All processing completed!"

# # To save the visualization instead of displaying
# python visualize_dino.py --npy_path "${DATASET_PATH}/dino/${RESOLUTION}x/000001.npy" --output_path visualization.png