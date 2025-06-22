#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/fs-computility/ai4phys/wangweida/workspace/LLaMA-Factory/saves/llama3-8b-it/new/gemma_molgen_all_raw  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH}