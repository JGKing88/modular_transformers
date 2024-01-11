#!/bin/bash
#SBATCH --job-name=MT_gpt2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=evlab
#SBATCH --mem=100G

source ~/.bashrc

module load openmind8/cuda/11.7
# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

MT_HOME="/om2/user/${USER_NAME}/modular_transformers"

conda activate modular_transformers
echo $(which python)

python "${MT_HOME}/modular_transformers/dynamics/time_is_layers.py"
