#!/bin/bash

#SBATCH --job-name=matt_job_guess
#SBATCH --output=logs/matt_job_guess.out.%j
#SBATCH --error=logs/matt_job_guess.out.%j
#SBATCH --time=48:00:00
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3

srun bash -c "hostname;"
srun bash -c "python models.py --train_guesser;"
