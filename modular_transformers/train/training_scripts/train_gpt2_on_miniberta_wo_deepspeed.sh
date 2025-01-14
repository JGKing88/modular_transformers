#!/bin/bash
#SBATCH --job-name=MT_gpt2
#SBATCH --time=6-0:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=evlab
#SBATCH --mem=50G

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

# accelerate launch --config_file "${MT_HOME}/modular_transformers/train/configs/deepspeed_config_A100.yaml" "${MT_HOME}/modular_transformers/train/extra_loss_train.py"

# accelerate launch --config_file "${MT_HOME}/modular_transformers/train/configs/accelerate_config.yaml" "${MT_HOME}/modular_transformers/train/extra_loss_train.py"
accelerate launch --config_file "${MT_HOME}/modular_transformers/train/configs/accelerate_config.yaml" "/om2/user/jackking/modular_transformers/scripts/training_straightness/extra_loss_train.py"

###accelerate launch --config_file "$/om2/user/jackking/modular_transformers/modular_transformers/train/configs/accelerate_config.yaml" "$/om2/user/jackking/modular_transformers/modular_transformers/train/extra_loss_train_test.py"


###srun -n 1 -t 01:00:00 --gres=gpu:a100:1 --mem=120G --pty bash  


#####SBATCH --mem=200G
#####SBATCH --nodelist=node105
#####SBATCH --exclude node017,node018
#####SBATCH --nodelist=node105
######SBATCH --gres=gpu:a100:2
#####SBATCH --gres=gpu:a100:2

######SBATCH --gres=gpu:A100:1

####SBATCH --gres=gpu:a100:2

########SBATCH --gres=gpu:1:2
####SBATCH --gres=gpu:a100:1

######SBATCH --gres=gpu:1 --constraint=high-capacity

### --config_file "${MT_HOME}/modular_transformers/train/configs/accelerate_config.yaml"

####accelerate launch --config_file /om/user/ehoseini/st/huggingface/accelerate/default_config.yaml "/om2/user/jackking/modular_transformers/modular_transformers/train/accelerate_train_gpt2.py"
