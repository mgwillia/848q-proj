#!/bin/bash

#SBATCH --job-name=matt_job_guess
#SBATCH --output=logs/matt_job_guess.out.%j
#SBATCH --error=logs/matt_job_guess.out.%j
#SBATCH --time=24:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load cuda/11.0.3

srun bash -c "hostname;"
srun bash -c "python models.py --split_rule last-a --train_guesser;"
