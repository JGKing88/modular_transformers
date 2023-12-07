#!/bin/bash
#SBATCH --job-name=MT_gpt2
#SBATCH --time=2-12:00:00
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
# run the .bash_profile file from USER_NAME home directory
# . /home/${USER_NAME}/.bash_profile

conda activate modular_transformers
echo $(which python)

accelerate launch --config_file "${MT_HOME}/modular_transformers/train/configs/accelerate_config.yaml" "${MT_HOME}/modular_transformers/train/accelerate_train_gpt2.py"

#### accelerate launch --config_file "configs/accelerate_config.yaml" "testing_script.py"


#####SBATCH --mem=200G
#####SBATCH --nodelist=node105
#####SBATCH --exclude node017,node018
#####SBATCH --nodelist=node105
######SBATCH --gres=gpu:a100:2
#####SBATCH --gres=gpu:a100:2

######SBATCH --gres=gpu:A100:1

####SBATCH --gres=gpu:a100:2

########SBATCH --gres=gpu:RTXA6000:2

######SBATCH --gres=gpu:1 --constraint=high-capacity

### --config_file "${MT_HOME}/modular_transformers/train/configs/accelerate_config.yaml"

####accelerate launch --config_file /om/user/ehoseini/st/huggingface/accelerate/default_config.yaml "/om2/user/jackking/modular_transformers/modular_transformers/train/accelerate_train_gpt2.py"
