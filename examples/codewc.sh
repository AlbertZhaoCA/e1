#!/bin/bash
#SBATCH -A aiscii
#SBATCH -p aiscii
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=256gb
#SBATCH --time=3-00:00:00
#SBATCH --job-name=entropy
#SBATCH --output=entropy_inter_ours.out
#SBATCH --error=entropy_sft_inter_ours.err
set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/codegenwc.yaml \
    worker.actor.model.model_path=${MODEL_PATH} 