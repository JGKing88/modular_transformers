#!/bin/bash
#SBATCH --job-name=MT_gpt2
#SBATCH --time=7-00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=evlab
#SBATCH --mem=20G

source ~/.bashrc

module load openmind8/cuda/11.7
# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

# run the .bash_profile file from USER_NAME home directory
# . /home/${USER_NAME}/.bash_profile
conda activate modular_transformers
echo $(which python)

python /om2/user/jackking/modular_transformers/train_rnn.py