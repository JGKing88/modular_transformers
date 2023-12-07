#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 --constraint=high-capacity
#SBATCH --ntasks=1
#SBATCH --mem=3G
#SBATCH -p evlab
#SBATCH -x node[100-116]

source ~/.bashrc

# module load openmind/cuda/11.3
module load openmind8/cuda/11.7
# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

MT_HOME="/om2/user/${USER_NAME}/modular_transformers/"
# run the .bash_profile file from USER_NAME home directory
#. /home/${USER_NAME}/.bash_profile

conda activate modular_transformers
echo $(which python)

accelerate launch --config_file "${MT_HOME}/modular_transformers/train/configs/deepspeed_config.yaml" "${MT_HOME}/modular_transformers/train/accelerate_train_gpt2.py"

