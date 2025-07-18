# import re
# from typing import Any, Dict, List
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from examples.reward_function.lexecute import run_scripts_with_timeout

# MARKDOWN_CODEBLOCK_RE = re.compile(r"^\s*```(?:python|Python|py)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

# def accuracy_reward(response: str, test_input: List[str], test_output: List[str]) -> Dict[str, float]:
#     response = MARKDOWN_CODEBLOCK_RE.sub("", response).strip()

#     try:
#         results = run_scripts_with_timeout(
#             script=response,
#             inputs=test_input,
#             time_limit=5.0,
#         )
#         if any("Timeout Error" in r or "<this is error>" in r.lower() for r in results if isinstance(r, str)):
#             print(f"[ERROR] Execution failed: {results[:10]}")
#         correct_count = sum(1 for res, expected in zip(results, test_output) if " ".join(res.split()) == " ".join(expected.split()))
#         acc = correct_count / len(test_output) if test_output else 0.0
#         print(f"[REWARD] Accuracy: {acc:.3f}")
#         return {"overall": acc, "accuracy": acc}

#     except Exception as e:
#         print(f"[ERROR] Execution failed: {e}")
#         return {"overall": 0.0, "accuracy": 0.0}


# def compute_score(
#     reward_inputs: List[Dict[str, Any]],
#     batch_size: int = 1024,
#     max_workers: int = 48
# ) -> List[Dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for math reward function.")

#     print("Computing scores in batches with parallel workers...")
#     scores = []

#     for i in range(0, len(reward_inputs), batch_size):
#         batch = reward_inputs[i:i + batch_size]

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = [
#                 executor.submit(
#                     accuracy_reward,
#                     r["response"], r["test_input"], r["test_output"]
#                 ) for r in batch
#             ]
#             for future in as_completed(futures):
#                 try:
#                     scores.append(future.result())
#                 except Exception as e:
#                     print(f"[ERROR] Future failed: {e}")
#                     scores.append({"overall": 0.0, "accuracy": 0.0})

#         print(f"Processed batch {i // batch_size + 1}/{(len(reward_inputs) + batch_size - 1) // batch_size}")

#     return scores

# import os
# import re
# import json
# import matplotlib.pyplot as plt
# from datetime import datetime
# from typing import Any, Dict, List
# from collections import Counter, defaultdict
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from examples.reward_function.lexecute import run_scripts_with_timeout

# MARKDOWN_CODEBLOCK_RE = re.compile(r"^\s*```(?:python|Python|py)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


# def accuracy_reward(response: str, test_input: List[str], test_output: List[str]) -> Dict[str, Any]:
#     response = MARKDOWN_CODEBLOCK_RE.sub("", response).strip()

#     try:
#         results = run_scripts_with_timeout(
#             script=response,
#             inputs=test_input,
#             time_limit=5.0,
#         )

#         if not isinstance(results, list):
#             results = [str(results)]

#         has_error = any(
#             isinstance(r, str) and (
#                 "timeout error" in r.lower()
#                 or "<this is error>" in r.lower()
#                 or "error" in r.lower()
#             ) for r in results
#         )

#         correct_count = sum(
#             1 for res, expected in zip(results, test_output)
#             if " ".join(res.split()) == " ".join(expected.split())
#         )
        
#         total = len(test_output)
#         acc = correct_count / total if total else 0.0


#         no_error_bonus = 0.1 if not has_error else 0.0
#         all_correct_bonus = 0.2 if correct_count == total and total > 0 else 0.0
#         overall = acc + no_error_bonus + all_correct_bonus
#         print("this is output:",results[:10],"this is expected:",test_output[:10],"this is acc:",acc)
#         return {
#             "overall": overall,
#             "accuracy": acc,
#             "raw_results": results,
#             "no_error_bonus": no_error_bonus,
#             "all_correct_bonus": all_correct_bonus
#         }

#     except Exception as e:
#         return {
#             "overall": 0.0,
#             "accuracy": 0.0,
#             "raw_results": [str(e)],
#             "no_error_bonus": 0.0,
#             "all_correct_bonus": 0.0
#         }


# def collect_debug_metrics(records: List[Dict[str, Any]]) -> None:
#     accuracies = []
#     overall_scores = []
#     error_types = Counter()
#     error_detail_counter = defaultdict(int)
#     no_error_bonus_count = 0
#     all_correct_bonus_count = 0

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_dir = os.path.join("debug_metrics", timestamp)
#     os.makedirs(save_dir, exist_ok=True)

#     table_data = []

#     for record in records:
#         question = record.get("question", "")
#         response = record.get("response", "")
#         test_input = record.get("test_input", [])
#         test_output = record.get("test_output", [])
#         result = record.get("result", {})

#         accuracy = result.get("accuracy", 0.0)
#         overall = result.get("overall", 0.0)
#         no_error_bonus = result.get("no_error_bonus", 0.0)
#         all_correct_bonus = result.get("all_correct_bonus", 0.0)
#         results = result.get("raw_results", [])

#         error_flag = False
#         if not isinstance(results, list):
#             error_types["non-list result"] += 1
#             error_flag = True
#         else:
#             for r in results:
#                 if isinstance(r, str) and "timeout error" in r.lower():
#                     error_types["timeout"] += 1
#                     error_flag = True
#                 elif isinstance(r, str) and "error" in r.lower():
#                     error_types["runtime error"] += 1
#                     error_detail_counter[r.strip().split(":")[0]] += 1
#                     error_flag = True

#         if not error_flag:
#             accuracies.append(accuracy)
#             overall_scores.append(overall)

#         if no_error_bonus > 0:
#             no_error_bonus_count += 1
#         if all_correct_bonus > 0:
#             all_correct_bonus_count += 1

#         table_data.append({
#             "question": question,
#             "response": response,
#             "test_input": test_input,
#             "test_output": test_output,
#             "accuracy": accuracy,
#             "overall": overall,
#             "no_error_bonus": no_error_bonus,
#             "all_correct_bonus": all_correct_bonus,
#             "results": results,
#             "errors_detected": error_flag
#         })

#     with open(os.path.join(save_dir, "debug_table.json"), "w", encoding="utf-8") as f:
#         json.dump(table_data, f, indent=2, ensure_ascii=False)

#     plt.figure(figsize=(8, 5))
#     plt.hist(overall_scores, bins=20, color="skyblue", edgecolor="black")
#     plt.title("Overall Score Distribution")
#     plt.xlabel("Overall Score")
#     plt.ylabel("Count")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "score_distribution.png"))
#     plt.close()

#     summary = {
#         "total_records": len(records),
#         "valid_records": len(overall_scores),
#         "timeout_count": error_types.get("timeout", 0),
#         "runtime_error_count": error_types.get("runtime error", 0),
#         "non_list_result_count": error_types.get("non-list result", 0),
#         "error_type_counts": dict(error_types),
#         "error_details": dict(error_detail_counter),
#         "no_error_bonus_count": no_error_bonus_count,
#         "all_correct_bonus_count": all_correct_bonus_count,
#     }

#     with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2, ensure_ascii=False)

#     print(f"[INFO] Metrics saved to: {save_dir}")


# def compute_score(
#     reward_inputs: List[Dict[str, Any]],
#     batch_size: int = 1024,
#     max_workers: int = 48
# ) -> List[Dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for math reward function.")

#     print("Computing scores in batches with parallel workers...")
#     all_records = []
#     scores = []

#     for i in range(0, len(reward_inputs), batch_size):
#         batch = reward_inputs[i:i + batch_size]

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = [
#                 executor.submit(
#                     accuracy_reward,
#                     r["response"], r["test_input"], r["test_output"]
#                 ) for r in batch
#             ]
#             for j, future in enumerate(as_completed(futures)):
#                 try:
#                     result = future.result()
#                 except Exception as e:
#                     print(f"[ERROR] Future failed: {e}")
#                     result = {
#                         "overall": 0.0,
#                         "accuracy": 0.0,
#                         "raw_results": [str(e)],
#                         "no_error_bonus": 0.0,
#                         "all_correct_bonus": 0.0
#                     }

#                 batch[j]["result"] = result
#                 all_records.append(batch[j])
#                 scores.append({
#                     "overall": result["overall"],
#                     "accuracy": result["accuracy"],
#                     "no_error_bonus": result["no_error_bonus"],
#                     "all_correct_bonus": result["all_correct_bonus"]
#                 })

#         print(f"Processed batch {i // batch_size + 1}/{(len(reward_inputs) + batch_size - 1) // batch_size}")

#     collect_debug_metrics(all_records)
#     return scores



# -*- coding: utf-8 -*-
"""
Reward‑function utilities with debug metrics.
Fix: ensure results correspond to the correct record when using ThreadPoolExecutor.
"""

# import os
# import re
# import json
# import matplotlib.pyplot as plt
# from datetime import datetime
# from typing import Any, Dict, List
# from collections import Counter, defaultdict
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from examples.reward_function.lexecute import run_scripts_with_timeout

# # ---------------------------  helpers  ---------------------------------------

# MARKDOWN_CODEBLOCK_RE = re.compile(
#     r"^\s*```(?:python|Python|py)?\s*|\s*```\s*$",
#     re.IGNORECASE | re.MULTILINE
# )

# def accuracy_reward(
#     response: str, test_input: List[str], test_output: List[str]
# ) -> Dict[str, Any]:
#     """
#     运行用户代码，比较输出，给出 accuracy + bonus。
#     返回 dict, 供 compute_score 聚合。
#     """
#     response = MARKDOWN_CODEBLOCK_RE.sub("", response).strip()

#     try:
#         results = run_scripts_with_timeout(
#             script=response,
#             inputs=test_input,
#             time_limit=5.0,
#         )

#         if not isinstance(results, list):
#             results = [str(results)]

#         has_error = any(
#             isinstance(r, str) and (
#                 "timeout error" in r.lower()
#                 or "<this is error>" in r.lower()
#                 or "error" in r.lower()
#             )
#             for r in results
#         )

#         correct_count = sum(
#             1 for res, expected in zip(results, test_output)
#             if " ".join(res.split()) == " ".join(expected.split())
#         )

#         total = len(test_output)
#         acc = correct_count / total if total else 0.0

#         no_error_bonus = 0.1 if not has_error else 0.0
#         all_correct_bonus = 0.2 if correct_count == total and total > 0 else 0.0
#         overall = acc + no_error_bonus + all_correct_bonus

#         return {
#             "overall": overall,
#             "accuracy": acc,
#             "raw_results": results,
#             "no_error_bonus": no_error_bonus,
#             "all_correct_bonus": all_correct_bonus
#         }

#     except Exception as e:
#         # 捕获运行时异常
#         return {
#             "overall": 0.0,
#             "accuracy": 0.0,
#             "raw_results": [str(e)],
#             "no_error_bonus": 0.0,
#             "all_correct_bonus": 0.0
#         }

# # -----------------------  debug helpers  -------------------------------------

# def collect_debug_metrics(records: List[Dict[str, Any]]) -> None:
#     """将整体评测结果保存为 json + 图表，方便排查问题。"""
#     accuracies, overall_scores = [], []
#     error_types = Counter()
#     error_detail_counter = defaultdict(int)
#     no_error_bonus_count = 0
#     all_correct_bonus_count = 0

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_dir = os.path.join("debug_metrics", timestamp)
#     os.makedirs(save_dir, exist_ok=True)

#     table_data = []

#     for record in records:
#         question = record.get("question", "")
#         response = record.get("response", "")
#         test_input = record.get("test_input", [])
#         test_output = record.get("test_output", [])
#         result = record.get("result", {})

#         accuracy = result.get("accuracy", 0.0)
#         overall = result.get("overall", 0.0)
#         no_error_bonus = result.get("no_error_bonus", 0.0)
#         all_correct_bonus = result.get("all_correct_bonus", 0.0)
#         results = result.get("raw_results", [])

#         error_flag = False
#         if not isinstance(results, list):
#             error_types["non-list result"] += 1
#             error_flag = True
#         else:
#             for r in results:
#                 if isinstance(r, str) and "timeout error" in r.lower():
#                     error_types["timeout"] += 1
#                     error_flag = True
#                 elif isinstance(r, str) and "error" in r.lower():
#                     error_types["runtime error"] += 1
#                     error_detail_counter[r.strip().split(":")[0]] += 1
#                     error_flag = True

#         if not error_flag:
#             accuracies.append(accuracy)
#             overall_scores.append(overall)

#         if no_error_bonus > 0:
#             no_error_bonus_count += 1
#         if all_correct_bonus > 0:
#             all_correct_bonus_count += 1

#         table_data.append({
#             "question": question,
#             "response": response,
#             "test_input": test_input,
#             "test_output": test_output,
#             "accuracy": accuracy,
#             "overall": overall,
#             "no_error_bonus": no_error_bonus,
#             "all_correct_bonus": all_correct_bonus,
#             "results": results,
#             "errors_detected": error_flag
#         })

#     # 保存明细
#     with open(os.path.join(save_dir, "debug_table.json"), "w", encoding="utf-8") as f:
#         json.dump(table_data, f, indent=2, ensure_ascii=False)

#     # 分布直方图
#     plt.figure(figsize=(8, 5))
#     plt.hist(overall_scores, bins=20, edgecolor="black")
#     plt.title("Overall Score Distribution")
#     plt.xlabel("Overall Score")
#     plt.ylabel("Count")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "score_distribution.png"))
#     plt.close()

#     # 汇总
#     summary = {
#         "total_records": len(records),
#         "valid_records": len(overall_scores),
#         "timeout_count": error_types.get("timeout", 0),
#         "runtime_error_count": error_types.get("runtime error", 0),
#         "non_list_result_count": error_types.get("non-list result", 0),
#         "error_type_counts": dict(error_types),
#         "error_details": dict(error_detail_counter),
#         "no_error_bonus_count": no_error_bonus_count,
#         "all_correct_bonus_count": all_correct_bonus_count,
#     }

#     with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2, ensure_ascii=False)

#     print(f"[INFO] Metrics saved to: {save_dir}")

# # -----------------------------  main  ----------------------------------------

# def compute_score(
#     reward_inputs: List[Dict[str, Any]],
#     batch_size: int = 1024,
#     max_workers: int = 48
# ) -> List[Dict[str, float]]:
#     """
#     并行计算整个评测集的得分。
#     修复点：确保 future 结果回填到正确的 record。
#     """
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for math reward function.")

#     print("Computing scores in batches with parallel workers...")
#     all_records: List[Dict[str, Any]] = []
#     scores: List[Dict[str, float]] = []

#     for i in range(0, len(reward_inputs), batch_size):
#         batch = reward_inputs[i : i + batch_size]

#         # 占位，用于保持与 batch 顺序一致
#         results_holder: List[Dict[str, Any]] = [None] * len(batch)

#         # 并行跑 accuracy_reward
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # future -> idx 映射，保证回填到正确位置
#             future_to_idx = {
#                 executor.submit(
#                     accuracy_reward,
#                     rec["response"], rec["test_input"], rec["test_output"]
#                 ): idx
#                 for idx, rec in enumerate(batch)
#             }

#             for future in as_completed(future_to_idx):
#                 idx = future_to_idx[future]
#                 try:
#                     res = future.result()
#                 except Exception as e:
#                     print(f"[ERROR] Future failed: {e}")
#                     res = {
#                         "overall": 0.0,
#                         "accuracy": 0.0,
#                         "raw_results": [str(e)],
#                         "no_error_bonus": 0.0,
#                         "all_correct_bonus": 0.0
#                     }
#                 results_holder[idx] = res

#         # 将结果写回对应 record，再做统计
#         for rec, res in zip(batch, results_holder):
#             rec["result"] = res
#             all_records.append(rec)
#             scores.append({
#                 "overall": res["overall"],
#                 "accuracy": res["accuracy"],
#                 "no_error_bonus": res["no_error_bonus"],
#                 "all_correct_bonus": res["all_correct_bonus"]
#             })

#         total_batches = (len(reward_inputs) + batch_size - 1) // batch_size
#         print(f"Processed batch {i // batch_size + 1}/{total_batches}")

#     collect_debug_metrics(all_records)
#     return scores


import os
import re
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Any, Dict, List
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from examples.reward_function.lexecute import run_scripts_with_timeout

# --------------------------- helpers -----------------------------

MARKDOWN_CODEBLOCK_RE = re.compile(
    r"^\s*```(?:python|Python|py)?\s*|\s*```\s*$",
    re.IGNORECASE | re.MULTILINE
)

def accuracy_reward(
    response: str, test_input: List[str], test_output: List[str]
) -> Dict[str, Any]:
    """
    运行用户代码，比较输出，给出 accuracy + bonus。
    返回 dict, 供 compute_score 聚合。
    """
    response = MARKDOWN_CODEBLOCK_RE.sub("", response).strip()

    try:
        results = run_scripts_with_timeout(
            script=response,
            inputs=test_input,
            time_limit=10.0,
        )
        
        if not isinstance(results, list):
            results = [str(results)]

        has_error = any(
            isinstance(r, str) and (
                "<this is error>" in r.lower()
            )
            for r in results
        )

        correct_count = sum(
            1 for res, expected in zip(results, test_output)
            if " ".join(res.split()) == " ".join(expected.split())
        )

        total = len(test_output)
        acc = correct_count / total if total else 0.0

        no_error_bonus = 0.1 if not has_error else 0.0
        all_correct_bonus = 0.2 if correct_count == total and total > 0 else 0.0
        overall = acc + no_error_bonus + all_correct_bonus

        return {
            "overall": overall,
            "accuracy": acc,
            "raw_results": results,
            "no_error_bonus": no_error_bonus,
            "all_correct_bonus": all_correct_bonus
        }

    except Exception as e:
        # 捕获运行时异常
        return {
            "overall": 0.0,
            "accuracy": 0.0,
            "raw_results": [str(e)],
            "no_error_bonus": 0.0,
            "all_correct_bonus": 0.0
        }

# -------------------- debug helpers ---------------------------

def collect_debug_metrics(records: List[Dict[str, Any]]) -> None:
    """将整体评测结果保存为 json + 图表，方便排查问题。"""
    accuracies, overall_scores = [], []
    error_types = Counter()
    error_detail_counter = defaultdict(int)
    no_error_bonus_count = 0
    all_correct_bonus_count = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("debug_metrics", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    table_data = []

    for record in records:
        question = record.get("question", "")
        response = record.get("response", "")
        test_input = record.get("test_input", [])
        test_output = record.get("test_output", [])
        result = record.get("result", {})

        accuracy = result.get("accuracy", 0.0)
        overall = result.get("overall", 0.0)
        no_error_bonus = result.get("no_error_bonus", 0.0)
        all_correct_bonus = result.get("all_correct_bonus", 0.0)
        results = result.get("raw_results", [])

        error_flag = False
        if not isinstance(results, list):
            error_types["non-list result"] += 1
            error_flag = True
        else:
            for r in results:
                if isinstance(r, str) and "timeout error" in r.lower():
                    error_types["timeout"] += 1
                    error_flag = True
                elif isinstance(r, str) and "error" in r.lower():
                    error_types["runtime error"] += 1
                    error_detail_counter[r.strip().split(":")[0]] += 1
                    error_flag = True

        if not error_flag:
            accuracies.append(accuracy)
            overall_scores.append(overall)

        if no_error_bonus > 0:
            no_error_bonus_count += 1
        if all_correct_bonus > 0:
            all_correct_bonus_count += 1

        table_data.append({
            "question": question,
            "response": response,
            "test_input": test_input,
            "test_output": test_output,
            "accuracy": accuracy,
            "overall": overall,
            "no_error_bonus": no_error_bonus,
            "all_correct_bonus": all_correct_bonus,
            "results": results,
            "errors_detected": error_flag
        })

    # 保存明细
    with open(os.path.join(save_dir, "debug_table.json"), "w", encoding="utf-8") as f:
        json.dump(table_data, f, indent=2, ensure_ascii=False)

    # 分布直方图
    plt.figure(figsize=(8, 5))
    plt.hist(overall_scores, bins=20, edgecolor="black")
    plt.title("Overall Score Distribution")
    plt.xlabel("Overall Score")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "score_distribution.png"))
    plt.close()

    # 汇总
    summary = {
        "total_records": len(records),
        "valid_records": len(overall_scores),
        "timeout_count": error_types.get("timeout", 0),
        "runtime_error_count": error_types.get("runtime error", 0),
        "non_list_result_count": error_types.get("non-list result", 0),
        "error_type_counts": dict(error_types),
        "error_details": dict(error_detail_counter),
        "no_error_bonus_count": no_error_bonus_count,
        "all_correct_bonus_count": all_correct_bonus_count,
    }

    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Metrics saved to: {save_dir}")

# -------------------------- main ----------------------------

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    batch_size: int = 512,
    max_workers: int = 65
) -> List[Dict[str, float]]:
    """
    并行计算整个评测集的得分。
    修复点：确保 future 结果回填到正确的 record。
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    print("Computing scores in batches with parallel workers...")
    all_records: List[Dict[str, Any]] = []
    scores: List[Dict[str, float]] = []

    for i in range(0, len(reward_inputs), batch_size):
        batch = reward_inputs[i : i + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda rec: accuracy_reward(rec["response"], rec["test_input"], rec["test_output"]), 
                batch
            ))

        for rec, res in zip(batch, results):
            rec["result"] = res
            all_records.append(rec)
            scores.append({
                "overall": res["overall"],
                "accuracy": res["accuracy"],
                "no_error_bonus": res["no_error_bonus"],
                "all_correct_bonus": res["all_correct_bonus"]
            })

        total_batches = (len(reward_inputs) + batch_size - 1) // batch_size
        print(f"Processed batch {i // batch_size + 1}/{total_batches}")

    # collect_debug_metrics(all_records)
    return scores
