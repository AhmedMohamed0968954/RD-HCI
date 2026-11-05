#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=30:00
#SBATCH --job-name=vsc_intro_install

cd $VSC_DATA

module load cluster/wice/gpu_p100

# Keep pip's download cache out of $VSC_HOME (3 GB quota); the CUDA wheels are several GB.
export PIP_CACHE_DIR="$VSC_DATA/.cache/pip"

conda update -n base -c defaults conda -y

# Create conda environment
conda create -n rd_lab python=3.12 -y

conda activate rd_lab

# Install required packages
pip install numpy matplotlib tqdm ipykernel
# The P100 is a Pascal GPU (sm_60). PyTorch's CUDA 12.8/12.9 wheels no longer contain Pascal
# kernels, so use the CUDA 12.6 wheels; PyTorch 2.14 is the last release that publishes them.
pip install "torch<2.15" torchvision --index-url https://download.pytorch.org/whl/cu126

# Test that CUDA kernels actually run on this GPU: is_available() alone can be True on an unsupported GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print('Kernel test:', (torch.ones(2, device='cuda') * 2).sum().item())"