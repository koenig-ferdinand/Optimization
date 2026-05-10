#!/bin/bash
#SBATCH --job-name=adamw_train
#SBATCH --account=es_he
#SBATCH --gpus=rtx_4090:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/scratch/leochen/Muon/V4/logs/slurm_adamw_%j.out
#SBATCH --error=/cluster/scratch/leochen/Muon/V4/logs/slurm_adamw_%j.err

eval "$(/cluster/scratch/leochen/miniconda3/bin/conda shell.bash hook)"
conda activate muon

cd /cluster/scratch/leochen/Muon/V4
mkdir -p logs

torchrun --standalone --nproc_per_node=4 train_adamw_prewarm_fine.py
