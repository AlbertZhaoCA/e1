# import re
# from typing import Any, Dict, List
# from examples.reward_function.execute import run_scripts_with_chunk,worker


# def accuracy_reward(response: str, test_input: list, test_output: list) -> float:
#     if not response or not test_input or not test_output:
#         return 0.0

#     results = run_scripts_with_chunk(
#         code_list=[response] * len(test_input),
#         test_input_list=test_input,
#         time_limit_list=[1] * len(test_input),
#         worker=worker,
#         num_chunks=1
#     )
#     print(results)


#     correct_count = sum(1 for res, expected in zip(results, test_output) if res == expected)
#     return correct_count / len(test_output) if test_output else 0.0
    

# def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for math reward function.")
    
#     scores = []
#     for reward_input in reward_inputs:
#         accuracy_score = accuracy_reward(reward_input["response"], reward_input["test_input"], reward_input["test_output"])
#         print(accuracy_score)
#         scores.append(
#             {
#                 "overall": accuracy_score,
#                 "accuracy": accuracy_score,
#             }
#         )

#     return scores

# from typing import Any, Dict, List
# from examples.reward_function.execute import run_scripts_with_timeout


# def accuracy_reward(response: str, test_input: list, test_output: list) -> float:
#     if not response or not test_input or not test_output:
#         return 0.0

#     results = run_scripts_with_timeout(
#         scripts=[response] * len(test_input),
#         inputs=test_input,
#         time_limit=0.1,
#     )


#     print(results)

#     correct_count = sum(1 for res, expected in zip(results, test_output) if res == expected)
#     return correct_count / len(test_output) if test_output else 0.0


# def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for math reward function.")

#     print('compute score!')
#     scores = []
#     for reward_input in reward_inputs:
#         acc_score = accuracy_reward(
#             reward_input["response"],
#             reward_input["test_input"],
#             reward_input["test_output"]
#         )
#         print(f"score: {acc_score}")
#         scores.append({
#             "overall": acc_score,
#             "accuracy": acc_score,
#         })

#     return scores

import re
import math
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from examples.reward_function.lexecute import run_scripts_with_timeout_with_usage

# 匹配 markdown 代码块标记
MARKDOWN_CODEBLOCK_RE = re.compile(r"^\s*```(?:python|Python|py)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

def usage_reward_exp(runtime: float, memory: float, alpha=3.0, beta=0.1, time_coef=0.1, memory_coef=0.1) -> Dict[str, float]:
    """
    使用负指数函数计算复杂度奖励：
    - runtime 奖励 = exp(-alpha * time) * time_coef
    - memory 奖励 = exp(-beta * memory) * memory_coef
    """
    time_score = math.exp(-alpha * runtime) * time_coef
    memory_score = math.exp(-beta * memory) * memory_coef
    return {"time_score": time_score, "memory_score": memory_score}

def accuracy_reward(response: str, test_input: List[str], test_output: List[str]) -> Dict[str, float]:
    response = MARKDOWN_CODEBLOCK_RE.sub("", response).strip()

    try:
        results = run_scripts_with_timeout_with_usage(
            script=response,
            inputs=test_input,
            time_limit=1.0,
        )
        print(results)
        correct_count = 0
        runtimes = []
        memories = []

        for res, expected in zip(results, test_output):
            output = res["output"]
            if "Timeout Error" in output or "<THIS IS Error>" in output.lower():
                print(f"[ERROR] Execution failed: {output}")
                continue

            if " ".join(output.split()) == " ".join(expected.split()):
                correct_count += 1

            runtimes.append(res["time"])
            memories.append(res["memory"])

        acc = correct_count / len(test_output) if test_output else 0.0

        time_score = 0.0
        memory_score = 0.0

        if acc == 1.0 and runtimes and memories:
            avg_runtime = sum(runtimes) / len(runtimes)
            avg_memory = sum(memories) / len(memories)
            usage = usage_reward_exp(avg_runtime, avg_memory)
            time_score = usage["time_score"]
            memory_score = usage["memory_score"]
            print("time_score:", time_score, "memory_score:", memory_score)

        overall = acc + time_score + memory_score
        print(f"[REWARD] Accuracy: {acc:.3f}, Time Score: {time_score:.3f}, Memory Score: {memory_score:.3f}, Overall: {overall:.3f}")

        return {
            "overall": overall,
            "accuracy": acc,
            "time_score": time_score,
            "memory_score": memory_score
        }

    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        return {
            "overall": 0.0,
            "accuracy": 0.0,
            "time_score": 0.0,
            "memory_score": 0.0
        }

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    batch_size: int = 1024,
    max_workers: int = 48
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for usage reward function.")

    print("Computing scores in batches with parallel workers...")
    scores = []

    for i in range(0, len(reward_inputs), batch_size):
        batch = reward_inputs[i:i + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    accuracy_reward,
                    r["response"], r["test_input"], r["test_output"]
                ) for r in batch
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(result)
                    scores.append(result)
                except Exception as e:
                    print(f"[ERROR] Future failed: {e}")
                    scores.append({
                        "overall": 0.0,
                        "accuracy": 0.0,
                        "time_score": 0.0,
                        "memory_score": 0.0
                    })

        print(f"Processed batch {i // batch_size + 1}/{(len(reward_inputs) + batch_size - 1) // batch_size}")

    return scores
