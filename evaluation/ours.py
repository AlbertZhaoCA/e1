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
#     "Gen-Verse/MBPP",
# ]

# AGENT1_API_URL = "http://localhost:8001/v1/chat/completions"
# AGENT1_MODEL = "/scratch/pioneer/jobs/job.2664465.hpc/models/saves/agent_higher_order_final/global_step_35/actor/huggingface"
# AGENT2_API_URL = "http://localhost:8001/v1/chat/completions"
# AGENT2_MODEL = "/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-7B-Instruct"

# def build_thought_prompt(problem: str) -> str:
#     return f"""You are an expert algorithm thinker. Given a problem, please provide a high-level strategy to solve it, without writing any code. Please ensure your reasoning is comprehensive and includes the following steps:

# Steps:
# 1. Define the **input-output structure**: What are the variables, types, and what does the function aim to accomplish?
# 2. Describe the **step-by-step solving process**:
#    - **Sequence**: List out the operations in the order they should be executed.
#    - **Branches**: Define any conditions (e.g., if / if-else) needed to handle different cases.
#    - **Loops**: What repetitive operations (for / while) are needed?
#    - **Edge Cases**: Mention if there are any potential edge cases that need special handling.

# Ensure that the reasoning is clear, logical, and without ambiguity.

# **IMPORTANT**: Only generate the reasoning, not code.

# Problem:
# {problem.strip()}
# """

# def build_code_prompt(problem: str, reasoning: str) -> str:
#     return f"""
# You are an expert algorithm coder. Given a problem and a high-level strategy, please implement the full solution in code.

# Problem:
# {problem.strip()}

# High-level Strategy:
# {reasoning.strip()}

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
#     try:
#         global AGENT1_API_URL, AGENT1_MODEL, AGENT2_API_URL, AGENT2_MODEL
#         reasoning = await call_api(AGENT1_API_URL, AGENT1_MODEL, build_thought_prompt(item["question"]), n=1)
#         prompt = build_code_prompt(item["question"], reasoning[0])
#         codes = await call_api(AGENT2_API_URL, AGENT2_MODEL, prompt, n=5)
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
#             "errors": []
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

#     for idx, code in enumerate(codes):
#         try:
#             start_time = time.time()
#             results = await asyncio.to_thread(
#                 run_scripts_with_timeout,
#                 script=code,
#                 inputs=item["test_input"],
#                 time_limit=5.0,
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
#         "errors": error_log
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

#     # Print error types and their proportions
#     total_errors = sum(error_types.values())
#     print(f"❌ Error Types Summary ({total_errors} total errors):")
#     for err, count in error_types.most_common():
#         error_percentage = (count / total_errors) * 100 if total_errors else 0.0
#         print(f"   - {err}: {count} ({error_percentage:.2f}%)")

#     return avg_acc

# async def main():
#     results = {}
#     for ds in DATASETS:
#         try:
#             acc = await evaluate_dataset(ds, AGENT1_MODEL, AGENT1_API_URL, batch_size=20)
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

MARKDOWN_CODEBLOCK_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.MULTILINE)

DATASETS = [
    "Gen-Verse/MBPP",
    "Gen-Verse/LiveBench",
    "Gen-Verse/CodeForces"
]

AGENT1_API_URL = "http://localhost:8000/v1/chat/completions"
AGENT1_MODEL = "/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-14B-Instruct"
AGENT2_API_URL = "http://localhost:8000/v1/chat/completions"
AGENT2_MODEL = "/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-14B-Instruct"

def build_thought_prompt(problem: str) -> str:
    return f"""You are an expert algorithm thinker. Given a problem, please provide a concise, high-level strategy without writing any code. 

Ensure your response is comprehensive and well-structured with clear steps and considerations.

Steps:
1. **Define the input-output structure**: Clearly specify the types and meanings of the input and output variables.
2. **Describe the solving logic** using:
   - **Sequence**: Step-by-step operations in the correct order.
   - **Branch**: Conditions (if / if-else) that lead to different solution paths.
   - **Loop**: Repetitive operations (for / while) needed to process the input.
3. **Key Considerations**: Identify any constraints, limitations, or potential pitfalls to keep in mind while solving the problem (e.g., time complexity, memory usage, edge cases, possilble errors to avoid). Provide solutions or optimizations where applicable.

Your response should provide a thorough strategy, avoid being overly concise, and include at least a few sentences for each step.

Example:
- **Input**:
  - **N** (an integer): The number of vertices in the graph.
  - **p** (a list of integers): A list of size N, where the i-th element, `p_i`, denotes the vertex that vertex i points to.
  
- **Output**:
  - Print "POSSIBLE" if a valid assignment exists, otherwise print "IMPOSSIBLE".

1. **Graph Construction**:
   1. Treat the given list `p` as a directed graph, where each vertex i points to `p_i`.
   2. The graph is essentially a **directed cycle structure** because each vertex has exactly one outgoing edge, forming a set of cycles.

2. **Identify Cycles**:
   1. Perform a Depth First Search (DFS) or Kosaraju’s algorithm to find the **strongly connected components (SCCs)** in the graph.
   2. Each SCC will either be a cycle or a group of vertices that form a cycle structure.

3. **Assigning Values**:
   1. For each SCC:
      1. Assign a **unique value** to each vertex in the SCC.
      2. Ensure that for each cycle, the vertices get distinct values. This is necessary for the condition that for each `x < a_i`, there exists a vertex `j` such that `a_j = x`.

4. **Check Feasibility**:
   1. After assigning values, verify that there are **disjoint cycles**.
   2. Ensure that there are no overlapping assignments between vertices belonging to different cycles.
   3. If any cycle is connected improperly or the structure is too intertwined, mark the assignment as **IMPOSSIBLE**.

5. **Return Result**:
   1. If all cycles are successfully assigned values, print **"POSSIBLE"**.
   2. If it is impossible to assign values (due to overlapping cycles or conflicting connections), print **"IMPOSSIBLE"**.

### Step 3: **Key Considerations**

1. **Time Complexity**:
   - The **SCC detection** using algorithms like Tarjan's or Kosaraju's runs in **O(N)** time.
   - Thus, the entire algorithm can be expected to run in **O(N)** time, which is optimal for the given problem constraints.

2. **Memory Usage**:
   - The space complexity primarily depends on the storage for the graph and the SCCs, leading to an **O(N)** space requirement.

3. **Edge Cases**:
   - The graph could consist of a single cycle (where every vertex points to exactly one other in a circular manner), which will be possible to assign distinct values.
   - If the graph has disconnected parts (not possible here since the graph is weakly connected), the solution might fail. But we are guaranteed weak connectivity, which simplifies the situation.

4. **Optimizations**:
   - Ensure that you can check cycles efficiently using union-find structures or DFS for SCCs.
   - Use efficient memory handling as the graph size can go up to 200,000 vertices.

5. **Possible Errors To Avoid**
   - ValueError: too many values to unpack:Ensure that each cycle or component is correctly unpacked or iterated over. For example, check the number of elements in a tuple or list before unpacking, and ensure that the number of values matches the expected structure.
   - IndexError: list index out of range: Carefully check that all vertex indices are within bounds. Ensure that the indices used to access the graph are valid (i.e., 0 <= i < N).

{problem.strip()}
"""



def build_code_prompt(problem: str, reasoning: str) -> str:
    return f"""
You are an expert algorithm coder. Given a problem and a high-level strategy, please implement the full solution in code.

Problem:
{problem.strip()}

High-level Strategy:
{reasoning.strip()}

Requirements:
- Do not Explain anything, wrap your code in this format ```python```
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
    """
    Checks whether the result matches the expected output for each test.
    """
    return [
        " ".join(r.strip().split()) == " ".join(e.strip().split())
        for r, e in zip(results, expected_outputs)
    ]

async def eval_one_sample(item: Dict[str, Any], model: str, api_url: str) -> Dict[str, Any]:
    code_generation_time = 0.0  # Initialize code generation time to 0
    execution_time = 0.0  # Initialize execution time to 0

    # Initialize error tracking variables
    failed_tests = 0
    total_tests = 0
    sample_error = False  # New flag to track if the sample has an error

    try:
        start_code_gen_time = time.perf_counter()
        # Get high-level reasoning from AGENT1
        reasoning = await call_api(AGENT1_API_URL, AGENT1_MODEL, build_thought_prompt(item["question"]), n=1)
        prompt = build_code_prompt(item["question"], reasoning[0])
        # Generate code with AGENT2
        codes = await call_api(AGENT2_API_URL, AGENT2_MODEL, prompt, n=5)
        code_generation_time = time.perf_counter() - start_code_gen_time
        
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
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
            "code_generation_time": code_generation_time,  # Returning initialized value
            "execution_time": execution_time
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

    for idx, code in enumerate(codes):
        try:
            start_exec_time = time.perf_counter()

            # Run the generated code
            results = await asyncio.to_thread(
                run_scripts_with_timeout,
                script=code,
                inputs=item["test_input"],
                time_limit=2.0,
            )
            execution_time += time.perf_counter() - start_exec_time

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

            total_tests += len(passes)
            failed_tests += len([p for p in passes if not p])

            for res in results:
                if isinstance(res, str) and res.startswith("<THIS IS Error>:"):
                    match = re.search(r"([A-Za-z_]+Error|[A-Za-z_]+Error.*|.*Error.*)", res)
                    if match:
                        error_log.append(match.group(1))
                    else:
                        error_log.append("UnknownError")
                    sample_error = True  # If any test has an error, mark the sample as erroneous

            correct = all(passes)
            accuracy_scores.append(1.0 if correct else 0.0)
            if correct:
                pass_5 = 1
        except Exception:
            accuracy_scores.append(0.0)
            cyclomatic_complexities.append(-1)
            unit_test_total += len(item["test_output"])

    pass_1 = 1 if accuracy_scores and accuracy_scores[0] == 1.0 else 0
    accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
    avg_cc = sum(c for c in cyclomatic_complexities if c >= 0) / len([c for c in cyclomatic_complexities if c >= 0]) if any(c >= 0 for c in cyclomatic_complexities) else -1
    avg_space = space_used / len(codes) if codes else 0
    avg_time = total_time / len(codes) if codes else 0

    
    error_rate_sample_based = 1 if sample_error else 0  # Based on samples (1 if any test failed in the sample)

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
        "error_rate_sample_based": error_rate_sample_based,
        "error": error_log[0] if error_log else "",
        "errors": error_log,
        "code_generation_time": code_generation_time,
        "execution_time": execution_time / len(codes) if codes else 0.0
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
    error_types = Counter()

    total_code_generation_time = 0.0
    total_execution_time = 0.0
    num_samples = 0

    for i in tqdm_asyncio(range(0, len(dataset), batch_size), desc="Batches", unit="batch"):
        batch = dataset[i : i + batch_size]
        batch_dicts = [dict(zip(dataset.column_names, vals)) for vals in zip(*batch.values())]
        batch_results = await asyncio.gather(*[eval_one_sample(item, model, api_url) for item in batch_dicts])
        all_scores.extend(batch_results)

        for result in batch_results:
            for err in result.get("errors", []):
                error_types[err] += 1

            # Accumulate the total times
            total_code_generation_time += result.get("code_generation_time", 0)
            total_execution_time += result.get("execution_time", 0)
            num_samples += 1

    avg_acc = sum(r["accuracy"] for r in all_scores) / len(all_scores)
    avg_time = sum(r["time"] for r in all_scores) / len(all_scores)
    avg_space = sum(r["space"] for r in all_scores) / len(all_scores)
    avg_cc = sum(r["cyclomatic_complexity"] for r in all_scores) / len(all_scores)
    pass1_rate = sum(r["pass_1"] for r in all_scores) / len(all_scores)
    pass5_rate = sum(r["pass_5"] for r in all_scores) / len(all_scores)
    total_unit = sum(r["unit_test_total"] for r in all_scores)
    passed_unit = sum(r["unit_test_passed"] for r in all_scores)
    unit_test_acc = passed_unit / total_unit if total_unit else 0.0

  

    # Calculate averages for code generation and execution times
    avg_code_generation_time = total_code_generation_time / num_samples if num_samples else 0
    avg_execution_time = total_execution_time / num_samples if num_samples else 0
    total_errors = sum(error_types.values())
    error_rate_sample = total_errors / total_unit
    error_rate_sample_based = sum(r["error_rate_sample_based"] for r in all_scores) / len(all_scores)
    print(f"✅ Finished evaluating {len(all_scores)} samples in {dataset_name}")
    print(f"📊 Average Accuracy: {avg_acc:.3f}")
    print(f"📐 Unit Test Accuracy: {unit_test_acc:.3f}")
    print(f"✅ Pass@1: {pass1_rate:.3f}, Pass@5: {pass5_rate:.3f}")
    print(f"🕒 Average Time: {avg_time:.3f}s | 📦 Average Space: {avg_space:.1f} chars | 🧠 Average CC: {avg_cc:.2f}")
    print(f"❌ Error Rate (per test): {error_rate_sample:.2%}")
    print(f"❌ Error Rate (per sample): {error_rate_sample_based:.2%}")

    # Print average code generation and execution time
    print(f"💻 Average Code Generation Time: {avg_code_generation_time:.3f}s")
    print(f"⏱ Average Execution Time: {avg_execution_time:.3f}s")

    print(f"❌ Error Types Summary ({total_errors} total errors):")
    for err, count in error_types.most_common(5):
        error_percentage = (count / total_errors) * 100 if total_errors else 0.0
        print(f"   - {err}: {count} ({error_percentage:.2f}%)")

    return avg_acc

async def main():
    results = {}
    for ds in DATASETS:
        try:
            acc = await evaluate_dataset(ds, AGENT1_MODEL, AGENT1_API_URL, batch_size=200)
            results[ds] = acc
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\u274c Error evaluating {ds}: {e}")
            results[ds] = None

    print("\U0001f3c1 Summary:")
    for ds, acc in results.items():
        print(f"- {ds}: {acc if acc is not None else 'Error'}")

if __name__ == "__main__":
    asyncio.run(main())
