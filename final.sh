#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=3090-gcondo --gres=gpu:1
#SBATCH --job-name final
#SBATCH --output final.out
#SBATCH --nodelist=gpu2252


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

#pip install kornia
#conda install -y -c conda-forge "setuptools<58.2.0"

which python
#conda install python=3.7.13

#conda install pip==22.3.1
#conda install -c conda-forge plyfile==0.8.1
#conda install tqdm

#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
#python -c "import torch; print(torch.cuda.is_available())"



#pip install submodules/depth-diff-gaussian-rasterization
#pip install submodules/simple-knn


#cd submodules/dqtorch
#python setup.py install
#cd ../..

#pip install torchmetrics

#pip install requests 

#pip install plotly
#pip install dash
#pip install scikit-learn
#pip install yaml
#pip install tensorboard 

#pip install scipy

#pip install kornia

#cd ~/data/yliang51/Gaussian4D/data
#pip install --upgrade --no-cache-dir gdown
#conda install -c conda-forge gdown
#gdown --id 1ibzV_4hOaQs8VF2X1ciYQ33jE9VOEVOW

#conda update libstdcxx-ng

export LD_LIBRARY_PATH=$CUDA_PREFIX/lib:$LD_LIBRARY_PATH

scenes="as basin bell cup plate press sieve"
for scene in $scenes; do

    python train_temporal.py -s ~/data/yliang51/Gaussian4D_depre/data/NeRF-DS/${scene}_novel_view --eval \
    	--posbase_pe=10 --timebase_pe=10 --defor_depth=6 --net_width=256 --use_skips \
        --model_path output/NeRF-DS/${scene}/full \
        --downsample 1 --sample_interval 1 \
        --fix_until_iter 3000 --init_mode_gaussian \
        --densify_until_iter 20_000 --opacity_reset_interval 3000 \
        --iterations=40000 --defor_lr_max_steps=30000 --position_lr_max_steps=30000 --scaling_lr_max_steps=30000 --rotation_lr_max_steps=40000 \
        --white_background \
        --stop_gradient \
        --l1_l2_switch 20000 --defor_lr 0.001\
        --num_pts 100000 \
        --enable_static \
        --disable_offopa --mult_quaternion
    


    python render_temporal.py  --eval \
    	--posbase_pe=10 --timebase_pe=10 --defor_depth=6 --net_width=256 --use_skips \
        --model_path output/NeRF-DS/${scene}/full \
        --downsample 1 \
        --white_background \
        --enable_static \
        --disable_offopa --mult_quaternion
    
done