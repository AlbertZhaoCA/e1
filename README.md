# EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework

## Requirements

### Software Requirements

- Python 3.9+
- transformers>=4.51.0
- flash-attn>=2.4.3
- vllm>=0.8.3


### Hardware Requirements

\* *estimated*

| Method                   | Bits |  1.5B  |   3B   |   7B   |   32B   |   72B   |
| ------------------------ | ---- | ------ | ------ | ------ | ------- | ------- |
| GRPO Full Fine-Tuning    |  AMP | 2*24GB | 4*40GB | 8*40GB | 16*80GB | 32*80GB |
| GRPO Full Fine-Tuning    | BF16 | 1*24GB | 1*40GB | 4*40GB |  8*80GB | 16*80GB |

> [!NOTE]
> Use `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16` to enable bf16 training.

### Train the Planner Agent

Before training the Planner agent, please launch the vLLM engine and update the URL and port in either `examples/reward_function/code.py` or `examples/reward_function/codewc.py`.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model path-to-yout-model \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --trust-remote-code
```   


```bash
bash examples/higher-order.sh
```

### Train the Coder Agent 

```bash
bash examples/codewoc.sh
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```
