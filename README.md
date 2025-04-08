# GauFRe: Gaussian Deformation Fields for Real-time Dynamic Novel View Synthesis

This repository is the official PyTorch implementation of the paper:

&nbsp;&nbsp;&nbsp;[**GauFRe: Gaussian Deformation Fields for Real-time Dynamic Novel View Synthesis**](https://lynl7130.github.io/gaufre/index.html)  
[Yiqing Liang](https://lynl7130.github.io)‡, [Numair Khan](https://nkhan2.github.io/), [Zhengqin Li](https://sites.google.com/a/eng.ucsd.edu/zhengqinli), [Thu Nguyen-Phuoc](https://www.monkeyoverflow.com/), [Douglas Lanman](https://www.linkedin.com/in/dlanman), [James Tompkin](https://jamestompkin.com/)‡, [Lei Xiao](https://leixiao-ubc.github.io/)  

<img width="20%" text-align="center" margin="auto" src=images/metalogo.png> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
‡ <img width="12%"  text-align="center" margin="auto" src=images/brownlogo.svg>

&nbsp;&nbsp;&nbsp;*WACV*, 2025 

&nbsp;&nbsp;&nbsp;[Paper](https://lynl7130.github.io/gaufre/static/pdfs/WACV_2025___GauFRe%20(1).pdf) / [Arxiv](https://arxiv.org/abs/2312.11458)

## Getting Started
This code has been developed with Anaconda (Python 3.9), CUDA 12.1.1 on Red Hat Enterprise Linux 9.2, one NVIDIA GeForce RTX 3090 GPU.  
Based on a fresh [Anaconda](https://www.anaconda.com/download/) environment ```gaufre```, following packages need to be installed:  

```Shell
conda create -n "gaufre" python=3.9
conda activate gaufre
conda install -c anaconda libstdcxx-ng
conda install -c menpo opencv 
conda install -c conda-forge plyfile==0.8.1
conda install nvidia/label/cuda-12.1.1::cuda-toolkit
pip install tqdm imageio

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())" # verify that torch is installed correctly

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
pip install lpips
pip install ftfy
pip install timm
pip install einops
pip install regex

# install from local folders 
cd submodules/dqtorch
python setup.py install
cd ../..
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/sam2

  ```

## Dataset

We follow the data organization of ["Monocular Dynamic Gaussian Splatting is Fast and Brittle but Smooth Motion Helps"](https://lynl7130.github.io/MonoDyGauBench.github.io/),
which could be downloaded [here](https://1drv.ms/f/c/4dd35d8ee847a247/EpmindtZTxxBiSjYVuaaiuUBr7w3nOzEl6GjrWjmVPuBFw?e=cW5gg1).

To use, one needs to unzip each ```[NestedPath]/[Scene].zip``` to be folder ```[NestedPath]/[Scene]```.


## Training and Inference

To train GauFRe on a scene ```[NestedPath]/[Scene]```, and save output to folder ```[OutputPath]```, 

```Shell
conda activate [YourPath]/gaufre
# for real-world scenes
bash scripts/trainval_real.sh [NestedPath]/[Scene] [OutputPath]
# for synthetic scenes
bash scripts/trainval_synthetic.sh [NestedPath]/[Scene] [OutputPath]
```



## Acknowledgement

Please cite our paper if you found our work useful:  

```bibtex
@inproceedings{liang2024gaufre,  
  Author = {Liang, Yiqing and Khan, Numair and Li, Zhengqin and Nguyen-Phuoc, Thu and Lanman, Douglas and Tompkin, James and Xiao, Lei},  
  Booktitle = {WACV},  
  Title = {GauFRe: Gaussian Deformation Fields for Real-time Dynamic Novel View Synthesis},  
  Year = {2025}  
}
```

- We thank ```https://github.com/graphdeco-inria/gaussian-splatting``` for source code reference.