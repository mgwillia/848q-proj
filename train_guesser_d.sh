#!/bin/bash

#SBATCH --job-name=split_rule_d
#SBATCH --output=logs/split_rule_d.out.%j
#SBATCH --error=logs/split_rule_d.out.%j
#SBATCH --time=24:00:00
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

module load cuda/11.0.3

srun bash -c "hostname;"
srun bash -c "python models.py --split_rule last-d --batch_size 32 --train_guesser;"
