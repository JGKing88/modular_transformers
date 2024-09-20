#!/bin/bash
#SBATCH -J jupyter
#SBATCH --time=2-00:00
#SBATCH -n 1
###SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
#SBATCH --mem 100G
###SBATCH --partition=evlab
#SBATCH -o jupyter.out
#SBATCH --exclude=node111
###SBATCH --exclude=node105

source ~/.bashrc

conda activate modular_transformers

unset XDG_RUNTIME_DIR

PORT=8081


jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --NotebookApp.allow_origin='*' --NotebookApp.port_retries=0