#!/bin/bash

#SBATCH --job-name=matt_eval
#SBATCH --output=logs/matt_eval.out.%j
#SBATCH --error=logs/matt_eval.out.%j
#SBATCH --time=24:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

module load cuda/11.0.3

srun bash -c "hostname;"
srun bash -c "python matt_eval.py --split_rule last-a;"
