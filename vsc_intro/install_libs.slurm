#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=15:00
#SBATCH --job-name=vsc_intro_install

cd $VSC_DATA

module load cluster/wice/gpu_p100

conda update -n base -c defaults conda -y

# Create conda environment
conda create -n rd_lab python=3.12 -y

conda activate rd_lab

# Install required packages
pip install numpy matplotlib tqdm ipykernel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Test torch cuda availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"