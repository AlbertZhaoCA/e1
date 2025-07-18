# DATASET="Gen-Verse/CodeContests"

# import asyncio
# import re
# import httpx
# from datasets import load_dataset
# from typing import List, Dict, Any
# from tqdm import tqdm
# from lexecute import run_scripts_with_timeout

# MARKDOWN_CODEBLOCK_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.MULTILINE)

# def build_thought_prompt(problem: str) -> str:
#     return f"""You are an expert algorithm thinker. Given a problem, please provide a high-level strategy without writing any code.

# Steps:
# 1. First, define the **input-output structure** clearly, including variable types and meanings.
# 2. Then, describe the **solving logic** using:
#    - **Sequence**: step-by-step operations
#    - **Branch**: conditions (if / if-else)
#    - **Loop**: repetitive logic (for / while)

# Each step must be concise and logically clear, avoiding ambiguity.

# **IMPORTANT**:
# - Only generate the solving process, not any code.
# - Your explanation should closely follow the given code, not guess a new solution.

# Example:

# Input: str: a string  
# Output: ch: a repeated character in str  

# 1: Initialize an empty dictionary `h`  
# 2: For each character `ch` in `str`:  
#    3: If `ch` is already in `h`:  
#       4: Return `ch`  
#    5: Else:  
#       6: Add `ch` to `h`  
# 7: Return None

# Problem:
# {problem.strip()}
# """

# def build_code_prompt(problem: str) -> str:
#     return f"""
# You are an expert algorithm coder. Given a problem description, implement the solution in Python.

# Requirements:
# - Do not explain anything.
# - Wrap your code in this format ```python```.
# - Use input() to input and print() to output in your script.

# Problem:
# {problem}


# """

# async def async_call_vllm(prompt: str) -> str:
#     async with httpx.AsyncClient(timeout=60.0) as client:
#         res = await client.post(
#             "http://localhost:8001/v1/chat/completions",
#             json={
#                 "model": "/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-7B-Instruct",
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.8,
#                 "top_p": 0.95,
#                 "top_k": 40,
#                 "min_p": 0.0,
#             }
#         )
#         data = res.json()
#         # 只取第一个生成结果
#         content = data["choices"][0]["message"]["content"]
#         return MARKDOWN_CODEBLOCK_RE.sub("", content).strip()

# async def eval_one_sample(item: Dict[str, Any]) -> Dict[str, float]:
#     code = await async_call_vllm(build_code_prompt(item["question"]))

#     try:
#         results = await asyncio.to_thread(
#             run_scripts_with_timeout,
#             script=code,
#             inputs=item["test_input"],
#             time_limit=1.0,
#         )
#         # ✅ 全对才算对
#         correct = all(
#             " ".join(r.strip().split()) == " ".join(e.strip().split())
#             for r, e in zip(results, item["test_output"])
#         )
#         score = 1.0 if correct else 0.0
#     except Exception:
#         score = 0.0

#     return {"accuracy": score, "score": score}

# async def evaluate_all(split="test", limit=None, batch_size=16):
#     dataset = load_dataset(DATASET, split=split)
#     if limit:
#         dataset = dataset.select(range(limit))

#     all_scores = []

#     for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
#         batch = dataset[i: i + batch_size]
#         batch = [dict(zip(dataset.column_names, vals)) for vals in zip(*batch.values())]
#         results = await asyncio.gather(*[eval_one_sample(item) for item in batch])
#         all_scores.extend(results)

#     return all_scores

# def main():
#     s = []
#     for i in range(10):
#         scores = asyncio.run(evaluate_all(split="test", limit=None, batch_size=100))
#         accs = [s["accuracy"] for s in scores]
#         avg_acc = sum(accs) / len(accs) if accs else 0.0
#         s.append(avg_acc)
#         print(f"\n✅ Finished evaluating {len(scores)} samples.")
#         print(f"📊 Average Accuracy: {avg_acc:.3f}")
#     print(f"📊 Average Accuracy over 5 runs: {sum(s)/len(s):.3f}")

# if __name__ == "__main__":
#     main()


# DATASET="Gen-Verse/CodeContests"

# import asyncio
# import re
# import httpx
# from datasets import load_dataset
# from typing import List, Dict, Any
# from tqdm import tqdm
# from lexecute import run_scripts_with_timeout

# MARKDOWN_CODEBLOCK_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.MULTILINE)

# def build_thought_prompt(problem: str) -> str:
#     return f"""You are an expert algorithm thinker. Given a problem, please provide a high-level strategy without writing any code.

# Steps:
# 1. First, define the **input-output structure** clearly, including variable types and meanings.
# 2. Then, describe the **solving logic** using:
#    - **Sequence**: step-by-step operations
#    - **Branch**: conditions (if / if-else)
#    - **Loop**: repetitive logic (for / while)

# Each step must be concise and logically clear, avoiding ambiguity.

# **IMPORTANT**:
# - Only generate the solving process, not any code.
# - Your explanation should closely follow the given code, not guess a new solution.

# Example:

# Input: str: a string  
# Output: ch: a repeated character in str  

# 1: Initialize an empty dictionary `h`  
# 2: For each character `ch` in `str`:  
#    3: If `ch` is already in `h`:  
#       4: Return `ch`  
#    5: Else:  
#       6: Add `ch` to `h`  
# 7: Return None

# Problem:
# {problem.strip()}
# """

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

# async def async_call_vllm_rl(prompt: str) -> str:
#     async with httpx.AsyncClient(timeout=60.0) as client:
#         res = await client.post(
#             "http://localhost:8001/v1/chat/completions",
#             json={
#                 "model": "/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-7B-Instruct",
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.2,
#                 "max_tokens": 512,

#             }
#         )
#         data = res.json()
#         # 只取第一个生成结果
#         content = data["choices"][0]["message"]["content"]
#         return MARKDOWN_CODEBLOCK_RE.sub("", content).strip()


# async def async_call_vllm(prompt: str) -> str:
#     async with httpx.AsyncClient(timeout=60.0) as client:
#         res = await client.post(
#             "http://localhost:8001/v1/chat/completions",
#             json={
#                 "model": "/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-7B-Instruct",
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.2,
#                 "top_p": 0.95,
#                 "top_k": 40,
#                 "min_p": 0.0,
#             }
#         )
#         data = res.json()
#         # 只取第一个生成结果
#         content = data["choices"][0]["message"]["content"]
#         return MARKDOWN_CODEBLOCK_RE.sub("", content).strip()

# async def eval_one_sample(item: Dict[str, Any]) -> Dict[str, float]:
#     reasoning = await async_call_vllm_rl(build_thought_prompt(item["question"]))
#     code = await async_call_vllm(build_code_prompt(item["question"], reasoning))
#     try:
#         results = await asyncio.to_thread(
#             run_scripts_with_timeout,
#             script=code,
#             inputs=item["test_input"],
#             time_limit=5.0,
#         )
#         total = len(item["test_output"])
#         correct = all(
#             " ".join(r.strip().split()) == " ".join(e.strip().split())
#             for r, e in zip(results, item["test_output"])
#         )
        
#     except Exception:
#         accuracy = 0.0

#     return {"accuracy": correct, "score": correct}

# async def evaluate_all(split="test", limit=None, batch_size=16):
#     dataset = load_dataset(DATASET, split=split)
#     if limit:
#         dataset = dataset.select(range(limit))

#     all_scores = []

#     for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
#         batch = dataset[i: i + batch_size]
#         batch = [dict(zip(dataset.column_names, vals)) for vals in zip(*batch.values())]
#         results = await asyncio.gather(*[eval_one_sample(item) for item in batch])
#         all_scores.extend(results)

#     return all_scores

# def main():
#     s = []
#     for i in range(10):
#         scores = asyncio.run(evaluate_all(split="test", limit=None, batch_size=100))
#         accs = [s["accuracy"] for s in scores]
#         avg_acc = sum(accs) / len(accs) if accs else 0.0
#         s.append(avg_acc)
#         print(f"\n✅ Finished evaluating {len(scores)} samples.")
#         print(f"📊 Average Accuracy: {avg_acc:.3f}")
#     print(f"📊 Average Accuracy over 5 runs: {sum(s)/len(s):.3f}")

# if __name__ == "__main__":
#     main()



import asyncio
import re
from typing import Dict, Any, List
from datasets import load_dataset
from tqdm import tqdm
from lexecute import run_scripts_with_timeout
import httpx

MARKDOWN_CODEBLOCK_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.MULTILINE)

DATASETS = [
    "Gen-Verse/CodeContests",
]

AGENT1_API_URL = "http://localhost:8000/v1/chat/completions"
AGENT1_MODEL = "/scratch/pioneer/jobs/job.2664465.hpc/sft/agent1"

AGENT2_API_URL = "http://localhost:8001/v1/chat/completions"
AGENT2_MODEL = "/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-7B-Instruct"

def build_thought_prompt(problem: str) -> str:
    return f"""You are an expert algorithm thinker. Given a problem, please provide a high-level strategy without writing any code.

Problem:
{problem.strip()}
"""

def build_code_prompt(problem: str, reasoning: str) -> str:
    return f"""
You are an expert algorithm coder. Given a problem, and a high-level strategy, please implement the full solution in code.

Problem:
{problem.strip()}

High-Level Strategy:
{reasoning.strip()}

Requirements:
- Do not Explain anything, wrap your code in this format ```python```
- **MOST IMPORTANT**: You should use input() to input and print() to output in your script.
"""

def clean_code(raw: str) -> str:
    return MARKDOWN_CODEBLOCK_RE.sub("", raw).strip()

async def call_api(url: str, model: str, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return clean_code(content)

async def eval_one_sample(item: Dict[str, Any]) -> Dict[str, float]:
    reasoning = await call_api(AGENT1_API_URL, AGENT1_MODEL, build_thought_prompt(item["question"]))
    code = await call_api(AGENT2_API_URL, AGENT2_MODEL, build_code_prompt(item["question"], reasoning))
    try:
        results = await asyncio.to_thread(
            run_scripts_with_timeout,
            script=code,
            inputs=item["test_input"],
            time_limit=1.0,
        )
        total = len(item["test_output"])
        correct = sum(
            1 for r, e in zip(results, item["test_output"])
            if " ".join(r.strip().split()) == " ".join(e.strip().split())
        )
        accuracy = correct / total if total else 0.0
    except Exception as e:
        print(f"❌ Error running code: {e}")
        accuracy = 0.0

    return {"accuracy": accuracy, "score": min(accuracy, 1.0)}

async def evaluate_dataset(dataset_name: str, limit: int = 100, batch_size: int = 16) -> float:
    print(f"🔍 Evaluating dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")

    def convert_dataset(ex):
        return {
            "question": ex.get("question", ex.get("prompt", "")),
            "test_input": ex.get("test_input", []),
            "test_output": ex.get("test_output", []),
        }
    dataset = dataset.map(convert_dataset)
    dataset = dataset.select(range(min(limit, len(dataset))))

    all_scores: List[Dict[str, float]] = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        batch_dicts = [dict(zip(dataset.column_names, vals)) for vals in zip(*batch.values())]
        batch_results = await asyncio.gather(*[eval_one_sample(item) for item in batch_dicts])
        all_scores.extend(batch_results)

    avg_acc = sum(r["accuracy"] for r in all_scores) / len(all_scores) if all_scores else 0.0
    print(f"✅ Finished evaluating {len(all_scores)} samples in {dataset_name}")
    print(f"📊 Average Accuracy: {avg_acc:.3f}\n")
    return avg_acc

async def main():
    results = {}
    for ds in DATASETS:
        try:
            acc = await evaluate_dataset(ds, limit=1000)
            results[ds] = acc
        except Exception as e:
            print(f"❌ Error evaluating {ds}: {e}")
            results[ds] = None

    print("🏁 Summary:")
    for ds, acc in results.items():
        print(f"- {ds}: {acc if acc is not None else 'Error'}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
