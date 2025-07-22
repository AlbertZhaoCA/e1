# AlgoForge: Specializing Code Generation Agents through Collaborative Reinforcement Learning

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

### 1. Train the Planner Agent

1. **Start the vLLM engine**
   Update the model name, URL and port in `examples/reward_function/higher-order.py` as needed, and verify the settings in `examples/higher-order.yaml`.

   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model path-to-your-model \
     --host 0.0.0.0 \
     --port 8000 \
     --tensor-parallel-size 4 \
     --trust-remote-code
   ```

2. **Launch the training script**

   ```bash
   bash examples/higher-order.sh
   ```

---

### 2. Train the Coder Agent


1. **Generate the dataset**
   Run `agent2_dataset.py`, then update the dataset path in either:

   * `examples/codegenwc.yaml`
   * `examples/codegenwoc.yaml`

2. **Launch the training script**

   ```bash
   bash examples/codewoc.sh
   ```

---

### 3. Merge Checkpoint to Hugging Face Format

Once training is complete, convert your checkpoint:

```bash
python3 scripts/model_merger.py \
  --local_dir to-the-actor-path-of-your-model-chekcpoint
```

---

Now you’re all set to run both agents end‑to‑end!
