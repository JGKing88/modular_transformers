#!/bin/bash
#SBATCH --job-name=MT_gpt2
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7-00:00
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
# run the .bash_profile file from USER_NAME home directory
# . /home/${USER_NAME}/.bash_profile

conda activate modular_transformers
echo $(which python)

accelerate launch --config_file "${MT_HOME}/modular_transformers/train/configs/accelerate_config.yaml" "${MT_HOME}/modular_transformers/train/mt_sweep.py"

#####accelerate launch --config_file "train/configs/accelerate_config.yaml" "train/mt_sweep.py"