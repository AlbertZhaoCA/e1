# import asyncio
# import re
# import httpx
# from typing import Any, Dict, List
# from examples.reward_function.lexecute import run_scripts_with_timeout  # 你的执行函数
# from tqdm import tqdm
# import numpy as np

# MARKDOWN_CODEBLOCK_RE = re.compile(r"^\s*```(?:python|Python|py)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


# def build_code_prompt(problem: str, reasoning: str) -> str:
#     return f"""
# You are an expert algorithm coder. Given a problem, and a high-level strategy, please implement the full solution in code.

# Problem:
# {problem.strip()}

# High-Level Strategy:
# {reasoning.strip()}

# Requirements:
# - Do not Explain anything, wrap your code in this format ```python```
# - **MOST IMPORTANT**: You should use input() to input and print() to output in your script.

# """




# async def async_call_vllm(prompt: str, n: int = 3) -> List[str]:
#     try:
#         async with httpx.AsyncClient() as client:
#             res = await client.post(
#                 "http://192.168.214.17:8000/v1/chat/completions",
#                 json={
#                     "model": "/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-7B-Instruct",
#                     "messages": [{"role": "user", "content": prompt}],
#                     "n": n,
#                     "temperature":1.0
#                 },
#                 timeout=60.0
#             )
#             choices = res.json()["choices"]
#             return [MARKDOWN_CODEBLOCK_RE.sub("", c["message"]["content"]).strip() for c in choices]
#     except Exception as e:
#         print(f"[ERROR] VLLM call failed: {e}")
#         return ["raise Exception('VLLM generation failed')"] * n


# async def async_accuracy_reward(reward_input: Dict[str, Any], n: int = 3) -> Dict[str, float]:
#     prompt = build_code_prompt(reward_input["question"], reward_input["response"])
#     code_candidates = await async_call_vllm(prompt, n=n)

#     scores = []

#     for i, code in enumerate(code_candidates):
#         try:
#             results = await asyncio.to_thread(
#                 run_scripts_with_timeout,
#                 script=code,
#                 inputs=reward_input["test_input"],
#                 time_limit=1.0,
#             )

#             total = len(reward_input["test_output"])
#             correct_count = 0
#             errors = 0

#             for res, expected in zip(results, reward_input["test_output"]):
#                 if "Timeout Error" in res or "error:" in res.lower():
#                     errors += 1
#                     continue

#                 if " ".join(res.split()) == " ".join(expected.split()):
#                     correct_count += 1

#             acc = correct_count / total if total > 0 else 0.0
#             scores.append(acc)

#         except Exception as e:
#             print(f"[WARNING] Evaluation failed for candidate {i}: {e}")
#             scores.append(0.0)

#     avg_score = sum(scores) / len(scores) if scores else 0.0
#     len_score = 0.1 if len(reward_input["response"]) < 320 else 0.0

#     final_score = min(avg_score + len_score, 1.0)


#     print(f"[REWARD] Average of {n} generations: {avg_score:.3f}, Final score: {final_score:.3f}", "Len score:", len_score)
#     return {"overall": final_score, "accuracy": avg_score, "len_score": len_score}


# def compute_score(reward_inputs: List[Dict[str, Any]], batch_size: int = 1024, n: int = 2) -> List[Dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for math reward function.")

#     async def run_all_batches():
#         all_scores = []
#         for i in tqdm(range(0, len(reward_inputs), batch_size), desc="Batches"):
#             batch = reward_inputs[i: i + batch_size]
#             tasks = [async_accuracy_reward(inp, n=n) for inp in batch]
#             results = await asyncio.gather(*tasks)
#             all_scores.extend(results)
#         return all_scores

#     return asyncio.run(run_all_batches())

import asyncio
import re
import httpx
from typing import Any, Dict, List
from examples.reward_function.lexecute import run_scripts_with_timeout
from tqdm import tqdm
import numpy as np

# 正则：去掉 markdown code block 包裹
MARKDOWN_CODEBLOCK_RE = re.compile(
    r"^\s*```(?:python|Python|py)?\s*|\s*```\s*$",
    re.IGNORECASE | re.MULTILINE
)

def build_code_prompt(problem: str, reasoning: str) -> str:
    return f"""
You are an expert algorithm coder. Given a problem, and a high-level strategy, please implement the full solution in code.

Problem:
{problem.strip()}

High-Level Strategy:
{reasoning.strip()}

Requirements:
- Do not explain anything, wrap your code in this format ```python```
- **MOST IMPORTANT**: You should use input() to input and print() to output in your script.
"""

async def async_call_vllm(prompt: str, n: int = 2) -> List[str]:
    """
    调用本地 VLLM 服务生成 n 个代码候选，去掉 markdown 包裹。
    """
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "http://192.168.214.13:8000/v1/chat/completions",
                json={
                    "model": "/scratch/pioneer/jobs/job.2664465.hpc/models/saves/agent_codegen/no_complexity/global_step_20/actor/huggingface",
                    "messages": [{"role": "user", "content": prompt}],
                    "n": n,
                    "temperature": 1.0
                },
                timeout=60.0
            )
            choices = res.json().get("choices", [])
            return [
                MARKDOWN_CODEBLOCK_RE.sub("", c["message"]["content"]).strip()
                for c in choices
            ]
    except Exception as e:
        print(f"[ERROR] VLLM call failed: {e}")
        return ["raise Exception('VLLM generation failed')"] * n

def length_reward(
    L: int,
    L_min: int = 300,
    L_max: int = 400,
    R_max: float = 0.1
) -> float:
    """
    三角形平滑长度奖励：
    - L 在 [L_min, L_max] 内：给满分 R_max
    - 之外按距离中点线性衰减
    """
    L_ref = (L_min + L_max) / 2
    if L_min <= L <= L_max:
        return R_max
    dist = abs(L - L_ref)
    scale = (L_max - L_min)
    return max(R_max * (1 - dist / scale), 0.0)

def accuracy_reward(
    scores: List[float],
    method: str = "softmax",
    T: float = 0.5
) -> float:
    """
    多种准确度聚合：
      - "simple": 普通平均
      - "harmonic": 调和平均（任何一条为 0 则整体为 0）
      - "softmax": softmax 加权平均
      - "mse": 均方平均
    """
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)

    if method == "harmonic":
        if any(s == 0 for s in arr):
            return 0.0
        n = len(arr)
        return n / np.sum(1.0 / arr)

    if method == "softmax":
        exps = np.exp(arr / T)
        weights = exps / np.sum(exps)
        return float(np.dot(weights, arr))

    if method == "mse":
        return float(np.mean(arr**2))

    # fallback to simple mean
    return float(np.mean(arr))

async def async_accuracy_reward(
    reward_input: Dict[str, Any],
    n: int = 2,
    acc_method: str = "softmax",
    T: float = 0.5
) -> Dict[str, float]:
    """
    对单条 question+response 进行打分：
      1. 调用 VLLM 生成 n 个代码候选
      2. 运行候选并对单元测试计算通过率列表
      3. 用 accuracy_reward 聚合通过率
      4. 用 length_reward 对 response 长度打分
      5. overall = min(acc + len_score, 1.0)
    返回字典：{"overall", "accuracy", "len_score"}
    """
    prompt = build_code_prompt(
        reward_input["question"],
        reward_input["response"]
    )
    code_candidates = await async_call_vllm(prompt, n=n)

    scores: List[float] = []
    error_free = True

    for i, code in enumerate(code_candidates):
        try:
            results = await asyncio.to_thread(
                run_scripts_with_timeout,
                script=code,
                inputs=reward_input["test_input"],
                time_limit=2.0,
            )
            total = len(reward_input["test_output"])
            correct = 0
            for out, exp in zip(results, reward_input["test_output"]):
                if "<THIS IS Error>" in out:
                    error_free = False
                    continue
                if " ".join(out.split()) == " ".join(exp.split()):
                    correct += 1
            scores.append(correct / total if total > 0 else 0.0)
        except Exception as e:
            print(f"[WARNING] Eval candidate {i} failed: {e}")
            scores.append(0.0)
            error_free = False

    avg_acc = accuracy_reward(scores, method=acc_method, T=T)
    no_error_bonus = 0.1 if error_free else 0.0
    overall = avg_acc + no_error_bonus
    return {"overall": overall, "accuracy": avg_acc,"no_error_bonus": no_error_bonus}

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    batch_size: int = 1024,
    n: int = 2,
    acc_method: str = "softmax",
    T: float = 0.5
) -> List[Dict[str, float]]:
    """
    对一批 reward_inputs 并行打分，默认 n=2 候选。
    每个 input 格式：
      {
        "question": str,
        "response": str,
        "test_input": [...],
        "test_output": [...]
      }
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please pass a list of reward_inputs.")

    async def _run_all():
        all_scores = []
        for i in tqdm(range(0, len(reward_inputs), batch_size), desc="Batches"):
            batch = reward_inputs[i : i + batch_size]
            tasks = [
                async_accuracy_reward(
                    inp, n=n, acc_method=acc_method, T=T
                )
                for inp in batch
            ]
            results = await asyncio.gather(*tasks)
            all_scores.extend(results)
        return all_scores

    return asyncio.run(_run_all())
