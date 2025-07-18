#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/codegenwoc.yaml \
    worker.actor.model.model_path=${MODEL_PATH} 