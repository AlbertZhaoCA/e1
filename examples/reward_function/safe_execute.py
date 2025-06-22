# import io
# import sys
# import multiprocessing
# from typing import List, Tuple

# sys.setrecursionlimit(500)

# def safe_worker(args: Tuple[str, str]) -> str:
#     script, input_val = args

#     input_lines = iter(input_val.splitlines())

#     def fake_input(prompt=""):
#         try:
#             return next(input_lines)
#         except StopIteration:
#             raise EOFError("No more input")

#     stdout_capture = io.StringIO()
#     original_stdout = sys.stdout
#     original_stdin = sys.stdin

#     sys.stdout = stdout_capture
#     sys.stdin = io.StringIO(input_val)

#     context = {
#         "__name__": "__main__",
#         "input": fake_input,
#     }

#     try:
#         exec(script, context)
#         return stdout_capture.getvalue()
#     except SystemExit:
#         return stdout_capture.getvalue()
#     except Exception as e:
#         return f"error: {e}"
#     finally:
#         sys.stdout = original_stdout
#         sys.stdin = original_stdin

# def run_scripts_with_pool(
#     scripts: List[str],
#     inputs: List[str],
#     time_limit: int = 1,
#     max_workers: int = None,
# ) -> List[str]:
#     results = []

#     with multiprocessing.Pool(processes=max_workers or multiprocessing.cpu_count()) as pool:
#         async_results = [
#             pool.apply_async(safe_worker, args=((scripts[i], inputs[i]),))
#             for i in range(len(scripts))
#         ]

#         for i, async_result in enumerate(async_results):
#             try:
#                 result = async_result.get(timeout=time_limit)
#             except multiprocessing.TimeoutError:
#                 result = "Timeout Error"
#             except Exception as e:
#                 result = f"Execution Error: {e}"
#             results.append(result)

#     return results

import ray
import io
import sys
from typing import List

# 设置最大递归深度（避免大代码段中嵌套过深）
sys.setrecursionlimit(500)

# === 安全执行 Python 脚本 ===
def safe_exec(script: str, input_val: str) -> str:
    input_lines = iter(input_val.splitlines())

    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")

    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin

    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)

    context = {
        "__name__": "__main__",
        "input": fake_input,
    }

    try:
        exec(script, context)
        return stdout_capture.getvalue()
    except Exception as e:
        return f"error: {e}"
    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin


# === 定义 Ray Actor，用于执行代码 ===
@ray.remote
class RewardExecutor:
    def run(self, script: str, input_val: str) -> str:
        return safe_exec(script, input_val)


# === 批量执行多个脚本 ===
def run_batch_with_actors(scripts: List[str], inputs: List[str], max_actors: int = 8) -> List[str]:
    assert len(scripts) == len(inputs)

    executors = [RewardExecutor.remote() for _ in range(max_actors)]
    futures = []

    for i, (script, input_val) in enumerate(zip(scripts, inputs)):
        executor = executors[i % max_actors]
        futures.append(executor.run.remote(script, input_val))

    results = ray.get(futures)
    return results

