#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/higher-order.yaml \
    worker.actor.model.model_path=${MODEL_PATH}