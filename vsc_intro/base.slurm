#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=15:00
#SBATCH --job-name=vsc_intro_training

module load cluster/wice/interactive

conda activate rd_lab

cd $VSC_DATA/vsc-intro

python training.py --visualize 
# You can use the arguments to change the comportment of the script, e.g.:
# python training.py --train eyes --epochs 20 --batch_size 64 --learning_rate 0.0005 --visualize --val 0.4
# will train the eye detection model for 20 epochs with a batch size of 64, a learning rate of 0.0005 and use 40% of the data for validation.