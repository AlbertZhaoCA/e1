import io
import os
import sys
import ast
import json
import time
import argparse
import multiprocessing
import sys
sys.setrecursionlimit(100)

####### execute the scripts with unit tests #########

def worker(script, input_val, output_queue):
    # Create an iterator over the input lines.
    input_lines = iter(input_val.splitlines())

    # Override the input() function in the exec context.
    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")
    
    # Redirect sys.stdout to capture printed output.
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin  # Save original stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)  # Simulate stdin with input_val

    context = {
        "__name__": "__main__",   # Ensures that `if __name__ == "__main__": ...` will fire
        "input": fake_input
    }

    try:
        exec(script, context)
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except SystemExit:
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except Exception as e:
        output_queue.put(f"error: {e}")

    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin


def run_scripts_with_timeout(scripts, inputs, time_limits, worker):
    results = [None] * len(scripts)
    processes = []
    queues = []
    deadlines = []

    for i in range(len(scripts)):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=worker, args=(scripts[i], inputs[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + time_limits[i])

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:
                results[i] = f"Execution Error: {e}"

    return results


def test_if_eq(x, y):
    return " ".join(x.split()) == " ".join(y.split())

def get_chunk_indices(n, num_chunks):
    chunk_size = n // num_chunks   
    remainder = n % num_chunks 
    indices = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        indices.append((start, end))
        start = end
    return indices

def run_scripts_with_chunk(code_list, test_input_list, time_limit_list, worker, num_chunks):

    chunks = get_chunk_indices(len(code_list), num_chunks)
    exe_results = []
    i = 0
    for start, end in chunks:
        sub_code_list = code_list[start:end]
        sub_test_input_list = test_input_list[start:end]
        sub_time_limit_list = time_limit_list[start:end]
        sub_exe_results = run_scripts_with_timeout(sub_code_list, sub_test_input_list, sub_time_limit_list, worker)
        exe_results = exe_results + sub_exe_results
        i += 1
    return exe_results

# import io
# import sys
# import ast
# import time
# from datetime import datetime
# import multiprocessing

# # 限制最大递归深度
# sys.setrecursionlimit(500)

# # 检查是否存在自递归函数或 lambda
# def has_dangerous_recursion(script: str) -> bool:
#     try:
#         tree = ast.parse(script)
#         function_defs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

#         # 检查函数是否递归调用自己
#         for func_name, func_node in function_defs.items():
#             for subnode in ast.walk(func_node):
#                 if isinstance(subnode, ast.Call):
#                     if hasattr(subnode.func, "id") and subnode.func.id == func_name:
#                         return True

#         # 检查是否包含 lambda 表达式（统一禁用）
#         for node in ast.walk(tree):
#             if isinstance(node, ast.Lambda):
#                 return True

#         return False
#     except Exception:
#         return True  # 如果无法解析，就认为是危险脚本

# # 安全执行脚本的 worker
# def worker(script, input_val, output_queue):
#     # 检查是否危险
#     if has_dangerous_recursion(script):
#         output_queue.put("error: Recursive function or lambda detected.")
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         with open(f"crash_script_{timestamp}.py", "w") as f:
#             f.write(script)
#         return

#     # 模拟输入
#     input_lines = iter(input_val.splitlines())

#     def fake_input(prompt=""):
#         try:
#             return next(input_lines)
#         except StopIteration:
#             raise EOFError("No more input")

#     # 重定向 stdout
#     stdout_capture = io.StringIO()
#     original_stdout = sys.stdout
#     original_stdin = sys.stdin
#     sys.stdout = stdout_capture
#     sys.stdin = io.StringIO(input_val)

#     context = {
#         "__name__": "__main__",
#         "input": fake_input
#     }

#     try:
#         exec(script, context)
#         printed_output = stdout_capture.getvalue()
#         output_queue.put(printed_output)

#     except SystemExit:
#         printed_output = stdout_capture.getvalue()
#         output_queue.put(printed_output)

#     except Exception as e:
#         output_queue.put(f"error: {e}")

#     finally:
#         sys.stdout = original_stdout
#         sys.stdin = original_stdin


# def run_scripts_with_timeout(scripts, inputs, time_limits, worker):
#     results = [None] * len(scripts)
#     processes = []
#     queues = []
#     deadlines = []

#     for i in range(len(scripts)):
#         q = multiprocessing.Queue()
#         p = multiprocessing.Process(target=worker, args=(scripts[i], inputs[i], q))
#         processes.append(p)
#         queues.append(q)
#         p.start()
#         deadlines.append(time.time() + time_limits[i])

#     while any(p.is_alive() for p in processes):
#         now = time.time()
#         for i, p in enumerate(processes):
#             if p.is_alive() and now >= deadlines[i]:
#                 p.terminate()
#                 results[i] = "Timeout Error"
#         time.sleep(0.001)

#     for i, p in enumerate(processes):
#         if results[i] is None:
#             try:
#                 results[i] = queues[i].get_nowait()
#             except Exception as e:
#                 results[i] = f"Execution Error: {e}"

#     return results


# def test_if_eq(x, y):
#     return " ".join(x.split()) == " ".join(y.split())

# def get_chunk_indices(n, num_chunks):
#     chunk_size = n // num_chunks   
#     remainder = n % num_chunks 
#     indices = []
#     start = 0
#     for i in range(num_chunks):
#         extra = 1 if i < remainder else 0
#         end = start + chunk_size + extra
#         indices.append((start, end))
#         start = end
#     return indices

# def run_scripts_with_chunk(code_list, test_input_list, time_limit_list, worker, num_chunks):

#     chunks = get_chunk_indices(len(code_list), num_chunks)
#     exe_results = []
#     i = 0
#     for start, end in chunks:
#         sub_code_list = code_list[start:end]
#         sub_test_input_list = test_input_list[start:end]
#         sub_time_limit_list = time_limit_list[start:end]
#         sub_exe_results = run_scripts_with_timeout(sub_code_list, sub_test_input_list, sub_time_limit_list, worker)
#         exe_results = exe_results + sub_exe_results
#         i += 1
#     return exe_results
