#!/bin/bash

# Job name
#SBATCH --job-name=MK-DATA

# Assign job to a queue
#SBATCH --partition=dgx

# Use GPU
#SBATCH --gres=gpu:1

# Assign job to a particular node
#SBATCH --nodelist=talos

# Assign RAM to job
#SBATCH --mem=50G

# Default configs for NGPU
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
export TFHUB_CACHE_DIR=.

# Activating conda enviroment
source ./env/bin/activate
python exmeshcnn/make_dataset $*

