#!/bin/bash

# Job name        |        |
#SBATCH --job-name=GRAD-CAM

# Assign job to a queue
#SBATCH --partition=dgx,dgx2

# Use GPU
#SBATCH --gres=gpu:1

# Assign job to a particular node
#SBATCH --nodelist=talos

# Default configs for NGPU
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
export TFHUB_CACHE_DIR=.

# Activating Python enviroment.
source ./env/bin/activate
# Calling the script passing all parameters from outside.
python ./exmeshcnn/Grad-CAM.py $*
