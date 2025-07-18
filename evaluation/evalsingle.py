# import asyncio
# import re
# import time
# from typing import Dict, Any, List
# from datasets import load_dataset
# from tqdm.asyncio import tqdm_asyncio
# import httpx
# from radon.complexity import cc_visit
# from collections import Counter
# from lexecute import run_scripts_with_timeout

# MARKDOWN_CODEBLOCK_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.MULTILINE)

# DATASETS = [
#     "Gen-Verse/CodeContests",
#     "Gen-Verse/MBPP",
#     "Gen-Verse/CodeForces",
#     "Gen-Verse/LiveBench",
#     "Gen-Verse/LiveCodeBench"
# ]
# AGENT1_API_URL = "http://localhost:8001/v1/chat/completions"
# AGENT1_MODEL = "/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-7B-Instruct"


# def build_code_prompt(problem: str) -> str:
#     return f"""
# You are an expert algorithm coder. Given a problem, and a high-level strategy, please implement the full solution in code.

# Problem:
# {problem.strip()}

# Requirements:
# - Do not Explain anything, wrap your code in this format ```python```
# - **MOST IMPORTANT**: You should use input() to input and print() to output in your script.
# """


# def clean_code(raw: str) -> str:
#     return MARKDOWN_CODEBLOCK_RE.sub("", raw).strip()


# async def call_api(url: str, model: str, prompt: str, n: int = 5) -> List[str]:
#     async with httpx.AsyncClient(timeout=60.0) as client:
#         response = await client.post(
#             url,
#             json={
#                 "model": model,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.2,
#                 "max_tokens": 512,
#                 "n": n,
#             },
#         )
#         response.raise_for_status()
#         data = response.json()
#         return [clean_code(choice["message"]["content"]) for choice in data["choices"]]


# def check_pass(results: List[str], expected_outputs: List[str]) -> List[bool]:
#     return [
#         " ".join(r.strip().split()) == " ".join(e.strip().split())
#         for r, e in zip(results, expected_outputs)
#     ]


# async def eval_one_sample(item: Dict[str, Any], model: str, api_url: str) -> Dict[str, Any]:
#     prompt = build_code_prompt(item["question"])
#     try:
#         codes = await call_api(api_url, model, prompt, n=5)
#     except Exception as e:
#         return {
#             "accuracy": 0.0,
#             "score": 0.0,
#             "pass_1": 0,
#             "pass_5": 0,
#             "time": 0.0,
#             "space": 0,
#             "cyclomatic_complexity": -1,
#             "unit_test_passed": 0,
#             "unit_test_total": 0,
#             "error": f"APIError: {str(e)}",
#         }

#     pass_1 = 0
#     pass_5 = 0
#     accuracy_scores = []
#     total_time = 0.0
#     space_used = 0
#     cyclomatic_complexities = []

#     unit_test_passed = 0
#     unit_test_total = 0

#     for idx, code in enumerate(codes):
#         try:
#             start_time = time.time()
#             results = await asyncio.to_thread(
#                 run_scripts_with_timeout,
#                 script=code,
#                 inputs=item["test_input"],
#                 time_limit=1.0,
#             )
#             duration = time.time() - start_time
#             total_time += duration

#             space_used += len(code)
#             try:
#                 cc_scores = cc_visit(code)
#                 cyclomatic_complexity = sum(block.complexity for block in cc_scores)
#             except Exception:
#                 cyclomatic_complexity = -1
#             cyclomatic_complexities.append(cyclomatic_complexity)

#             passes = check_pass(results, item["test_output"])
#             unit_test_passed += sum(passes)
#             unit_test_total += len(passes)

#             correct = all(passes)
#             accuracy_scores.append(1.0 if correct else 0.0)
#             if correct:
#                 pass_5 = 1
#         except Exception:
#             accuracy_scores.append(0.0)
#             cyclomatic_complexities.append(-1)
#             unit_test_total += len(item["test_output"])

#     pass_1 = 1 if accuracy_scores and accuracy_scores[0] == 1.0 else 0
#     accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
#     avg_cc = (
#         sum(c for c in cyclomatic_complexities if c >= 0) / len([c for c in cyclomatic_complexities if c >= 0])
#         if any(c >= 0 for c in cyclomatic_complexities)
#         else -1
#     )
#     avg_space = space_used / len(codes) if codes else 0
#     avg_time = total_time / len(codes) if codes else 0

#     return {
#         "accuracy": accuracy,
#         "score": min(accuracy, 1.0),
#         "pass_1": pass_1,
#         "pass_5": pass_5,
#         "time": avg_time,
#         "space": avg_space,
#         "cyclomatic_complexity": avg_cc,
#         "unit_test_passed": unit_test_passed,
#         "unit_test_total": unit_test_total,
#         "error": "",
#     }


# async def evaluate_dataset(dataset_name: str, model: str, api_url: str, limit: int = 1000, batch_size: int = 200) -> float:
#     print(f"🔍 Evaluating dataset: {dataset_name}")
#     dataset = load_dataset(dataset_name, split="test")

#     def convert_dataset(ex):
#         return {
#             "question": ex.get("question", ex.get("prompt", "")),
#             "test_input": ex.get("test_input", []),
#             "test_output": ex.get("test_output", []),
#         }

#     dataset = dataset.map(convert_dataset)
#     dataset = dataset.select(range(min(limit, len(dataset))))

#     all_scores: List[Dict[str, Any]] = []

#     for i in tqdm_asyncio(range(0, len(dataset), batch_size), desc="Batches", unit="batch"):
#         batch = dataset[i : i + batch_size]
#         batch_dicts = [dict(zip(dataset.column_names, vals)) for vals in zip(*batch.values())]
#         batch_results = await asyncio.gather(*[eval_one_sample(item, model, api_url) for item in batch_dicts])
#         all_scores.extend(batch_results)

#     avg_acc = sum(r["accuracy"] for r in all_scores) / len(all_scores)
#     avg_time = sum(r["time"] for r in all_scores) / len(all_scores)
#     avg_space = sum(r["space"] for r in all_scores) / len(all_scores)
#     avg_cc = sum(r["cyclomatic_complexity"] for r in all_scores) / len(all_scores)
#     pass1_rate = sum(r["pass_1"] for r in all_scores) / len(all_scores)
#     pass5_rate = sum(r["pass_5"] for r in all_scores) / len(all_scores)
#     total_unit = sum(r["unit_test_total"] for r in all_scores)
#     passed_unit = sum(r["unit_test_passed"] for r in all_scores)
#     unit_test_acc = passed_unit / total_unit if total_unit else 0.0

#     error_types = Counter()
#     for r in all_scores:
#         if r["error"]:
#             err_type = r["error"].split(":")[0].strip()
#             error_types[err_type] += 1

#     print(f"✅ Finished evaluating {len(all_scores)} samples in {dataset_name}")
#     print(f"📊 Average Accuracy: {avg_acc:.3f}")
#     print(f"🧪 Unit Test Accuracy: {unit_test_acc:.3f}")
#     print(f"⏱️ Average Time: {avg_time:.3f}s")
#     print(f"📦 Average Space: {avg_space:.1f} chars")
#     print(f"🧠 Average Cyclomatic Complexity: {avg_cc:.2f}")
#     print(f"✅ Pass@1: {pass1_rate:.3f}, Pass@5: {pass5_rate:.3f}")
#     print(f"❌ Error Summary ({sum(error_types.values())} total):")
#     for err, count in error_types.most_common():
#         print(f"   - {err}: {count}")
#     print()

#     return avg_acc


# async def main():
#     results = {}
#     for ds in DATASETS:
#         try:
#             acc = await evaluate_dataset(ds, AGENT1_MODEL, AGENT1_API_URL, batch_size=200)
#             results[ds] = acc
#         except Exception as e:
#             print(f"❌ Error evaluating {ds}: {e}")
#             results[ds] = None

#     print("🏁 Summary:")
#     for ds, acc in results.items():
#         print(f"- {ds}: {acc if acc is not None else 'Error'}")


# if __name__ == "__main__":
#     asyncio.run(main())

# import asyncio
# import re
# import time
# from typing import Dict, Any, List
# from datasets import load_dataset
# from tqdm.asyncio import tqdm_asyncio
# import httpx
# from radon.complexity import cc_visit
# from collections import Counter
# from lexecute import run_scripts_with_timeout

# MARKDOWN_CODEBLOCK_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.MULTILINE)

# DATASETS = [
#     "Gen-Verse/CodeContests",
#     "Gen-Verse/MBPP",
#     "Gen-Verse/CodeForces",
#     "Gen-Verse/LiveBench",
# ]

# AGENT1_API_URL = "http://localhost:8000/v1/chat/completions"
# AGENT1_MODEL = "/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-14B-Instruct"

# # You are an intelligent agent skilled at step‑by‑step reasoning and writing high‑quality code.
# # You are an expert algorithm coder. Given a problem, please implement the full solution in code.
# def build_code_prompt(problem: str) -> str:
#     return f"""
# You are an expert algorithm coder. Given a problem, please implement the full solution in code.
# Problem:
# {problem.strip()}

# Requirements:
# - Do not explain anything, wrap your code in this format ```python```
# - **MOST IMPORTANT**: You should use input() to input and print() to output in your script.
# """

# def clean_code(raw: str) -> str:
#     return MARKDOWN_CODEBLOCK_RE.sub("", raw).strip()

# async def call_api(url: str, model: str, prompt: str, n: int = 5) -> List[str]:
#     async with httpx.AsyncClient(timeout=60.0) as client:
#         response = await client.post(
#             url,
#             json={
#                 "model": model,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.2,
#                 "max_tokens": 512,
#                 "n": n,
#             },
#         )
#         response.raise_for_status()
#         data = response.json()
#         return [clean_code(choice["message"]["content"]) for choice in data["choices"]]

# def check_pass(results: List[str], expected_outputs: List[str]) -> List[bool]:
#     return [
#         " ".join(r.strip().split()) == " ".join(e.strip().split())
#         for r, e in zip(results, expected_outputs)
#     ]

# async def eval_one_sample(item: Dict[str, Any], model: str, api_url: str) -> Dict[str, Any]:
#     prompt = build_code_prompt(item["question"])
#     try:
#         # Measure code generation time using perf_counter
#         start_time = time.perf_counter()
#         codes = await call_api(api_url, model, prompt, n=5)
#         code_generation_time = time.perf_counter() - start_time  # Track code generation time

#     except Exception as e:
#         return {
#             "accuracy": 0.0,
#             "score": 0.0,
#             "pass_1": 0,
#             "pass_5": 0,
#             "time": 0.0,
#             "space": 0,
#             "cyclomatic_complexity": -1,
#             "unit_test_passed": 0,
#             "unit_test_total": 0,
#             "error": f"APIError: {str(e)}",
#             "errors": [],
#             "code_generation_time": 0.0,
#             "execution_time": 0.0
#         }

#     pass_1 = 0
#     pass_5 = 0
#     accuracy_scores = []
#     total_time = 0.0
#     space_used = 0
#     cyclomatic_complexities = []
#     unit_test_passed = 0
#     unit_test_total = 0
#     error_log = []
#     total_tests = 0  # Track total tests for error rate
#     failed_tests = 0  # Track failed tests for error rate
#     execution_time = 0.0  # Track execution time

#     for idx, code in enumerate(codes):
#         try:
#             # Measure execution time using perf_counter
#             start_execution_time = time.perf_counter()

#             results = await asyncio.to_thread(
#                 run_scripts_with_timeout,
#                 script=code,
#                 inputs=item["test_input"],
#                 time_limit=2.0,
#             )

#             execution_time += time.perf_counter() - start_execution_time  # Track execution time
#             total_time += execution_time
#             space_used += len(code)

#             try:
#                 cc_scores = cc_visit(code)
#                 cyclomatic_complexity = sum(block.complexity for block in cc_scores)
#             except Exception:
#                 cyclomatic_complexity = -1
#             cyclomatic_complexities.append(cyclomatic_complexity)

#             passes = check_pass(results, item["test_output"])
#             unit_test_passed += sum(passes)
#             unit_test_total += len(passes)

#             # Track the total and failed tests for error rate calculation
#             total_tests += len(passes)
#             failed_tests += len([p for p in passes if not p])

#             for res in results:
#                 if isinstance(res, str) and res.startswith("<THIS IS Error>:"):
#                     match = re.search(r"([A-Za-z_]+Error|[A-Za-z_]+Error.*|.*Error.*)", res)
#                     if match:
#                         error_log.append(match.group(1))
#                     else:
#                         error_log.append("UnknownError")

#             correct = all(passes)
#             accuracy_scores.append(1.0 if correct else 0.0)
#             if correct:
#                 pass_5 = 1
#         except Exception:
#             accuracy_scores.append(0.0)
#             cyclomatic_complexities.append(-1)
#             unit_test_total += len(item["test_output"])

#     pass_1 = 1 if accuracy_scores and accuracy_scores[0] == 1.0 else 0
#     accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
#     avg_cc = (
#         sum(c for c in cyclomatic_complexities if c >= 0) / len([c for c in cyclomatic_complexities if c >= 0])
#         if any(c >= 0 for c in cyclomatic_complexities)
#         else -1
#     )
#     avg_space = space_used / len(codes) if codes else 0
#     avg_time = total_time / len(codes) if codes else 0

#     # Calculate error rates:
#     error_rate_sample = (failed_tests / total_tests) if total_tests else 0.0
#     error_rate_sample_based = 1 if failed_tests > 0 else 0  # Based on samples (1 if any test failed in the sample)

#     return {
#         "accuracy": accuracy,
#         "score": min(accuracy, 1.0),
#         "pass_1": pass_1,
#         "pass_5": pass_5,
#         "time": avg_time,
#         "space": avg_space,
#         "cyclomatic_complexity": avg_cc,
#         "unit_test_passed": unit_test_passed,
#         "unit_test_total": unit_test_total,
#         "error_rate_sample": error_rate_sample,  # Error rate based on tests
#         "error_rate_sample_based": error_rate_sample_based,  # Error rate based on samples
#         "error": error_log[0] if error_log else "",
#         "errors": error_log,
#         "code_generation_time": code_generation_time,
#         "execution_time": execution_time / len(codes) if codes else 0.0
#     }

# async def evaluate_dataset(dataset_name: str, model: str, api_url: str, limit: int = 1000, batch_size: int = 200) -> float:
#     print(f"🔍 Evaluating dataset: {dataset_name}")
#     dataset = load_dataset(dataset_name, split="test")

#     def convert_dataset(ex):
#         return {
#             "question": ex.get("question", ex.get("prompt", "")),
#             "test_input": ex.get("test_input", []),
#             "test_output": ex.get("test_output", []),
#         }

#     dataset = dataset.map(convert_dataset)
#     dataset = dataset.select(range(min(limit, len(dataset))))

#     all_scores: List[Dict[str, Any]] = []

#     # Track errors separately
#     error_types = Counter()

#     for i in tqdm_asyncio(range(0, len(dataset), batch_size), desc="Batches", unit="batch"):
#         batch = dataset[i : i + batch_size]
#         batch_dicts = [dict(zip(dataset.column_names, vals)) for vals in zip(*batch.values())]
#         batch_results = await asyncio.gather(*[eval_one_sample(item, model, api_url) for item in batch_dicts])
#         all_scores.extend(batch_results)

#         # Track errors encountered in the batch
#         for result in batch_results:
#             for err in result.get("errors", []):
#                 error_types[err] += 1

#     avg_acc = sum(r["accuracy"] for r in all_scores) / len(all_scores)
#     avg_time = sum(r["time"] for r in all_scores) / len(all_scores)
#     avg_space = sum(r["space"] for r in all_scores) / len(all_scores)
#     avg_cc = sum(r["cyclomatic_complexity"] for r in all_scores) / len(all_scores)
#     pass1_rate = sum(r["pass_1"] for r in all_scores) / len(all_scores)
#     pass5_rate = sum(r["pass_5"] for r in all_scores) / len(all_scores)
#     total_unit = sum(r["unit_test_total"] for r in all_scores)
#     passed_unit = sum(r["unit_test_passed"] for r in all_scores)
#     unit_test_acc = passed_unit / total_unit if total_unit else 0.0

#     # Calculate overall error rates for the dataset
#     error_rate_sample = sum(r["error_rate_sample"] for r in all_scores) / len(all_scores)
#     error_rate_sample_based = sum(r["error_rate_sample_based"] for r in all_scores) / len(all_scores)

#     print(f"✅ Finished evaluating {len(all_scores)} samples in {dataset_name}")
#     print(f"📊 Average Accuracy: {avg_acc:.3f}")
#     print(f"📐 Unit Test Accuracy: {unit_test_acc:.3f}")
#     print(f"✅ Pass@1: {pass1_rate:.3f}, Pass@5: {pass5_rate:.3f}")
#     print(f"🕒 Time: {avg_time:.3f}s | 📦 Space: {avg_space:.1f} chars | 🧠 CC: {avg_cc:.2f}")
#     print(f"❌ Error Rate (per test): {error_rate_sample:.2%}")
#     print(f"❌ Error Rate (per sample): {error_rate_sample_based:.2%}")

#     # Print error types and their proportions (top 5 errors)
#     total_errors = sum(error_types.values())
#     print(f"❌ Error Types Summary ({total_errors} total errors):")
#     for err, count in error_types.most_common(5):
#         error_percentage = (count / total_errors) * 100 if total_errors else 0.0
#         print(f"   - {err}: {count} ({error_percentage:.2f}%)")

#     # Print average code generation and execution time
#     avg_code_gen_time = sum(r["code_generation_time"] for r in all_scores) / len(all_scores)
#     avg_execution_time = sum(r["execution_time"] for r in all_scores) / len(all_scores)
#     print(f"💻 Average Code Generation Time: {avg_code_gen_time:.3f}s")
#     print(f"⏱️ Average Execution Time: {avg_execution_time:.3f}s")

#     return avg_acc

# async def main():
#     results = {}
#     for ds in DATASETS:
#         try:
#             acc = await evaluate_dataset(ds, AGENT1_MODEL, AGENT1_API_URL, batch_size=200)
#             results[ds] = acc
#         except Exception as e:
#             print(f"\u274c Error evaluating {ds}: {e}")
#             results[ds] = None

#     print("\U0001f3c1 Summary:")
#     for ds, acc in results.items():
#         print(f"- {ds}: {acc if acc is not None else 'Error'}")

# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
import re
import time
from typing import Dict, Any, List
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
import httpx
from radon.complexity import cc_visit
from collections import Counter
from lexecute import run_scripts_with_timeout

MARKDOWN_CODEBLOCK_RE = re.compile(r'^[\s\S]*?```(?:python)?|```[\s\S]*$', re.DOTALL)


DATASETS = [
    "Gen-Verse/CodeContests",
    "Gen-Verse/MBPP",
    "Gen-Verse/LiveBench",
    "Gen-Verse/CodeForces"
]

AGENT1_API_URL = "http://localhost:8000/v1/chat/completions"
AGENT1_MODEL = "/scratch/pioneer/jobs/job.2664465.hpc/models/deepseek-coder-33b-instruct"

def build_code_prompt(problem: str) -> str:
    return f"""
You are an expert algorithm coder. Given a problem, please implement the full solution in code.
Problem:
{problem.strip()}

Requirements:
- Do not explain anything, wrap your code in this format ```python```
- **MOST IMPORTANT**: You should use input() to input and print() to output in your script.
"""

def clean_code(raw: str) -> str:
    return MARKDOWN_CODEBLOCK_RE.sub("", raw).strip()

async def call_api(url: str, model: str, prompt: str, n: int = 5) -> List[str]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 512,
                "n": n,
            },
        )
        response.raise_for_status()
        data = response.json()
        return [clean_code(choice["message"]["content"]) for choice in data["choices"]]

def check_pass(results: List[str], expected_outputs: List[str]) -> List[bool]:
    return [
        " ".join(r.strip().split()) == " ".join(e.strip().split())
        for r, e in zip(results, expected_outputs)
    ]

async def eval_one_sample(item: Dict[str, Any], model: str, api_url: str) -> Dict[str, Any]:
    prompt = build_code_prompt(item["question"])
    try:
        # Measure code generation time using perf_counter
        start_time = time.perf_counter()
        codes = await call_api(api_url, model, prompt, n=5)
        code_generation_time = time.perf_counter() - start_time  # Track code generation time

    except Exception as e:
        return {
            "accuracy": 0.0,
            "score": 0.0,
            "pass_1": 0,
            "pass_5": 0,
            "time": 0.0,
            "space": 0,
            "cyclomatic_complexity": -1,
            "unit_test_passed": 0,
            "unit_test_total": 0,
            "error": f"APIError: {str(e)}",
            "errors": [],
            "code_generation_time": 0.0,
            "execution_time": 0.0,
            "error_occurred": 1,  # Indicate an error occurred for this sample
            "error_rate_sample_based": 1,  # This sample has an error
            "error_rate_test_based": 1,  # Error for all tests in this sample
        }

    pass_1 = 0
    pass_5 = 0
    accuracy_scores = []
    total_time = 0.0
    space_used = 0
    cyclomatic_complexities = []
    unit_test_passed = 0
    unit_test_total = 0
    error_log = []
    total_tests = 0  # Track total tests for error rate
    failed_tests = 0  # Track failed tests for error rate
    execution_time = 0.0  # Track execution time
    error_occurred = 0  # Track if an error occurred in this sample
    error_rate_sample_based = 0  # Track per sample error rate
    error_rate_test_based = 0  # Track per test error rate

    for idx, code in enumerate(codes):
        try:
            # Measure execution time using perf_counter
            start_execution_time = time.perf_counter()

            results = await asyncio.to_thread(
                run_scripts_with_timeout,
                script=code,
                inputs=item["test_input"],
                time_limit=2.0,
            )

            execution_time += time.perf_counter() - start_execution_time  # Track execution time
            total_time += execution_time
            space_used += len(code)

            try:
                cc_scores = cc_visit(code)
                cyclomatic_complexity = sum(block.complexity for block in cc_scores)
            except Exception:
                cyclomatic_complexity = -1
            cyclomatic_complexities.append(cyclomatic_complexity)

            passes = check_pass(results, item["test_output"])
            unit_test_passed += sum(passes)
            unit_test_total += len(passes)

            # Track the total and failed tests for error rate calculation
            total_tests += len(passes)
            failed_tests += len([p for p in passes if not p])

            # Check for errors in results
            for res in results:
                if isinstance(res, str) and res.startswith("<THIS IS Error>:"):
                    error_rate_test_based = 1  # Error in at least one test
                    match = re.search(r"([A-Za-z_]+Error|[A-Za-z_]+Error.*|.*Error.*)", res)
                    if match:
                        error_log.append(match.group(1))
                    else:
                        error_log.append("UnknownError")
                    error_occurred = 1  # Mark error occurred for this sample

            correct = all(passes)
            accuracy_scores.append(1.0 if correct else 0.0)
            if correct:
                pass_5 = 1
        except Exception:
            accuracy_scores.append(0.0)
            cyclomatic_complexities.append(-1)
            unit_test_total += len(item["test_output"])
            error_occurred = 1  # Mark error occurred in this sample
            error_rate_test_based = 1  # Mark error for all tests in this sample

    pass_1 = 1 if accuracy_scores and accuracy_scores[0] == 1.0 else 0
    accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
    avg_cc = (
        sum(c for c in cyclomatic_complexities if c >= 0) / len([c for c in cyclomatic_complexities if c >= 0])
        if any(c >= 0 for c in cyclomatic_complexities)
        else -1
    )
    avg_space = space_used / len(codes) if codes else 0
    avg_time = total_time / len(codes) if codes else 0

    error_rate_sample_based = 1 if error_occurred else 0  # Based on samples (1 if any test failed in the sample)

    return {
        "accuracy": accuracy,
        "score": min(accuracy, 1.0),
        "pass_1": pass_1,
        "pass_5": pass_5,
        "time": avg_time,
        "space": avg_space,
        "cyclomatic_complexity": avg_cc,
        "unit_test_passed": unit_test_passed,
        "unit_test_total": unit_test_total,
        "error_rate_sample_based": error_rate_sample_based,  # Error rate based on samples
        "error_rate_test_based": error_rate_test_based,  # Error rate for tests
        "error": error_log[0] if error_log else "",
        "errors": error_log,
        "code_generation_time": code_generation_time,
        "execution_time": execution_time / len(codes) if codes else 0.0,
        "error_occurred": error_occurred  # Track if an error occurred
    }

async def evaluate_dataset(dataset_name: str, model: str, api_url: str, limit: int = 1000, batch_size: int = 200) -> float:
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

    all_scores: List[Dict[str, Any]] = []

    # Track errors separately
    error_types = Counter()
    error_samples = 0  # Track how many samples had an error

    for i in tqdm_asyncio(range(0, len(dataset), batch_size), desc="Batches", unit="batch"):
        batch = dataset[i : i + batch_size]
        batch_dicts = [dict(zip(dataset.column_names, vals)) for vals in zip(*batch.values())]
        batch_results = await asyncio.gather(*[eval_one_sample(item, model, api_url) for item in batch_dicts])
        all_scores.extend(batch_results)
        for result in batch_results:
            for err in result.get("errors", []):
                error_types[err] += 1


    avg_acc = sum(r["accuracy"] for r in all_scores) / len(all_scores)
    avg_time = sum(r["time"] for r in all_scores) / len(all_scores)
    avg_space = sum(r["space"] for r in all_scores) / len(all_scores)
    avg_cc = sum(r["cyclomatic_complexity"] for r in all_scores) / len(all_scores)
    pass1_rate = sum(r["pass_1"] for r in all_scores) / len(all_scores)
    pass5_rate = sum(r["pass_5"] for r in all_scores) / len(all_scores)
    total_unit = sum(r["unit_test_total"] for r in all_scores)
    passed_unit = sum(r["unit_test_passed"] for r in all_scores)
    unit_test_acc = passed_unit / total_unit if total_unit else 0.0
    # Print error types and their proportions (top 5 errors)
    total_errors = sum(error_types.values())
   

    # Calculate overall error rates for the dataset
    error_rate_sample_based = sum(r["error_rate_sample_based"] for r in all_scores) / len(all_scores)
    error_rate_sample = total_errors / total_unit

    print(f"✅ Finished evaluating {len(all_scores)} samples in {dataset_name}")
    print(f"📊 Average Accuracy: {avg_acc:.3f}")
    print(f"📐 Unit Test Accuracy: {unit_test_acc:.3f}")
    print(f"✅ Pass@1: {pass1_rate:.3f}, Pass@5: {pass5_rate:.3f}")
    print(f"🕒 Time: {avg_time:.3f}s | 📦 Space: {avg_space:.1f} chars | 🧠 CC: {avg_cc:.2f}")
    print(f"❌ Error Rate (per test): {error_rate_sample:.2%}")
    print(f"❌ Error Rate (per sample): {error_rate_sample_based:.2%}")

    print(f"❌ Error Types Summary ({total_errors} total errors):")
    for err, count in error_types.most_common(5):
        error_percentage = (count / total_errors) * 100 if total_errors else 0.0
        print(f"   - {err}: {count} ({error_percentage:.2f}%)")
    # Print average code generation and execution time
    avg_code_gen_time = sum(r["code_generation_time"] for r in all_scores) / len(all_scores)
    avg_execution_time = sum(r["execution_time"] for r in all_scores) / len(all_scores)
    print(f"💻 Average Code Generation Time: {avg_code_gen_time:.3f}s")
    print(f"⏱️ Average Execution Time: {avg_execution_time:.3f}s")

    return avg_acc

async def main():
    results = {}
    for ds in DATASETS:
        try:
            acc = await evaluate_dataset(ds, AGENT1_MODEL, AGENT1_API_URL, batch_size=200)
            results[ds] = acc
        except Exception as e:
            print(f"\u274c Error evaluating {ds}: {e}")
            results[ds] = None

    print("\U0001f3c1 Summary:")
    for ds, acc in results.items():
        print(f"- {ds}: {acc if acc is not None else 'Error'}")

if __name__ == "__main__":
    asyncio.run(main())
