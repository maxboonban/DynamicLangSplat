#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=3090-gcondo --gres=gpu:1
#SBATCH --job-name install
#SBATCH --output install.out




module load gcc/10.1.0-mojgbn
module load cmake/3.26.3-xi6h36u
module load cuda/12.1.1-ebglvvq
module load cudnn/8.9.6.50-12-56zgdoa
module load glm/0.9.9.8-m3s6sze

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh


cd ~/data/yliang51/Gaussian4D/gaussian-splatting-main

conda -V

#conda remove -p ~/data/yliang51/envs/gaufre --all
#conda create -p ~/data/yliang51/envs/gaufre python=3.9
conda activate ~/data/yliang51/envs/gaufre

which python 
which pip

conda install -c anaconda libstdcxx-ng
conda install -c menpo opencv 
python -c "import cv2"

conda install -c conda-forge plyfile==0.8.1
pip install tqdm imageio

pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"

pip install matplotlib

pip install torchmetrics

pip install requests 
pip install plotly
pip install dash
pip install scikit-learn
pip install yaml
pip install tensorboard 

pip install scipy

pip install kornia


pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/simple-knn



cd submodules/dqtorch
python setup.py install
cd ../..

pip install lpips
#cd ~/data/yliang51/Gaussian4D/data
#pip install --upgrade --no-cache-dir gdown
#conda install -c conda-forge gdown
#gdown --id 1ibzV_4hOaQs8VF2X1ciYQ33jE9VOEVOW



