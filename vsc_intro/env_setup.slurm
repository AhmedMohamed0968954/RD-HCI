#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=15:00
#SBATCH --job-name=vsc_intro_setup

cd $VSC_DATA

module load cluster/wice/gpu_p100


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $VSC_DATA/Miniconda3-latest-Linux-x86_64.sh
# It is important to use $VSC_DATA to store your conda installation since 
# environments tend to be large, and your quota in $VSC_HOME would be exceeded soon.
bash Miniconda3-latest-Linux-x86_64.sh -b -p $VSC_DATA/miniconda3

# Add line in .bashrc to put conda in PATH
echo 'export PATH="$VSC_DATA/miniconda3/bin:$PATH"' >> ~/.bashrc

export PATH="$VSC_DATA/miniconda3/bin:$PATH"

conda tos accept

# Update conda
conda update -n base -c defaults conda -y

conda init bash