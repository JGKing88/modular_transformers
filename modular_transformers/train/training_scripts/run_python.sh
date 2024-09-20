#!/bin/bash
#SBATCH --job-name=MT_gpt2
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
###SBATCH --gres=gpu:RTXA6000:1

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
# run the .bash_profile file from USER_NAME home directory
# . /home/${USER_NAME}/.bash_profile

conda activate modular_transformers
echo $(which python)

python "${MT_HOME}/scripts/training_straightness/curvature_analysis.py"
# python "${MT_HOME}/scripts/adding_straightness/perturb_straight.py"
# python "${MT_HOME}/scripts/adding_straightness/perturb_straight_by_act_replacement.py"
# python "${MT_HOME}/scripts/adding_straightness/calculate_surprisals.py"
# python "${MT_HOME}/scripts/adding_straightness/compare_to_gpt4.py"


