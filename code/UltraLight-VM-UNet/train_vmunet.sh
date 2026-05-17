#!/bin/bash
#SBATCH --job-name=vmunet_train
#SBATCH --partition=gpu3090      
#SBATCH --qos=4gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=vmunet_%j.log


export PYTHONPATH=$PYTHONPATH:$(pwd)


/gpfs/work/bio/yixuanli2204/.conda/envs/fyp/bin/python3 -u train.py