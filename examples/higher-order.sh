#!/bin/bash
#SBATCH -A aiscii
#SBATCH -p aiscii
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=256gb
#SBATCH --time=3-00:00:00
#SBATCH --job-name=planner
#SBATCH --output=planner.out
#SBATCH --error=planner.err

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-7B-Instruct

python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    > vllm.log 2>&1 &

VLLM_PID=$!

echo "vLLM started PID=${VLLM_PID}"

########################################
until curl -s http://localhost:8001/v1/models > /dev/null; do
  echo "waiting vllm..."
  sleep 5
done


python3 -m verl.trainer.main \
    config=examples/higher-order.yaml \
    worker.actor.model.model_path=${MODEL_PATH}