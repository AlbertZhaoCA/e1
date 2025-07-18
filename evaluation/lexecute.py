# import io
# import sys
# import multiprocessing
# import time
# import resource



# def worker(script, input_val, output_queue):
#     sys.setrecursionlimit(100)
#     max_memory = 256 * 1024 * 1024  # 256
#     resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
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
#         output_queue.put(stdout_capture.getvalue())
#     except SystemExit:
#         output_queue.put(stdout_capture.getvalue())
#     except Exception as e:
#         output_queue.put(f"error: {e}")
#     finally:
#         sys.stdout = original_stdout
#         sys.stdin = original_stdin


# def run_scripts_with_timeout(script, inputs, time_limit, worker):
#     results = [None] * len(inputs)
#     processes = []
#     queues = []

#     for i, input_val in enumerate(inputs):
#         q = multiprocessing.Queue()
#         p = multiprocessing.Process(target=worker, args=(script, input_val, q))
#         p.start()
#         processes.append(p)
#         queues.append(q)

#     deadlines = [time.time() + time_limit] * len(inputs)

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

# script = """
# class UnionFind:
#     def __init__(self, n):
#         self.parent = list(range(n))
#         self.rank = [0] * n

#     def find(self, u):
#         if self.parent[u] != u:
#             self.parent[u] = self.find(self.parent[u])
#         return self.parent[u]

#     def union(self, u, v):
#         pu, pv = self.find(u), self.find(v)
#         if pu == pv:
#             return False
#         if self.rank[pu] > self.rank[pv]:
#             self.parent[pv] = pu
#         elif self.rank[pv] > self.rank[pu]:
#             self.parent[pu] = pv
#         else:
#             self.parent[pv] = pu
#             self.rank[pu] += 1
#         return True

# def kruskal(n, edges):
#     mst_cost = 0
#     uf = UnionFind(n)
#     edges.sort(key=lambda x: x[2])
#     for u, v, cost in edges:
#         if uf.union(u, v):
#             mst_cost += cost
#     return mst_cost

# if __name__ == "__main__":
#     N = int(input())
#     points = []
#     for _ in range(N):
#         x, y = map(int, input().split())
#         points.append((x, y))

#     edges = []
#     for i in range(N):
#         for j in range(i + 1, N):
#             cost = min(abs(points[i][0] - points[j][0]), abs(points[i][1] - points[j][1]))
#             edges.append((i, j, cost))

#     print(kruskal(N, edges))
# """


# # 输入是一行 "1 2 3\n"
# inputs = ["4\n0 0\n2 2\n3 10\n5 2\n"]

# # 调用你给的函数
# results = run_scripts_with_timeout(script, inputs, time_limit=0.01, worker=worker)

# # 输出结果
# for i, res in enumerate(results):
#     print(f"Input {i+1}: {inputs[i].strip()} => Output: {res.strip()}")

import subprocess
import tempfile
import os

def run_scripts_with_timeout(script, inputs, time_limit):
    results = []

    for input_val in inputs:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_script_file:
                tmp_script_file.write(script)
                tmp_script_path = tmp_script_file.name

            result = subprocess.run(
                ["python3", tmp_script_path],
                input=input_val,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=time_limit,
            )

            if result.returncode == 0:
                results.append(result.stdout.strip())
            else:
                error_message = result.stderr.strip()
                if "Traceback" in error_message:  # Check if it's a Python traceback
                    error_lines = error_message.splitlines()
                    error_type_line = error_lines[-1]  # The last line should contain the exception type
                    results.append(f"<THIS IS Error>: {error_type_line}")
                else:
                    results.append(f"<THIS IS Error>: {error_message}")

        except subprocess.TimeoutExpired:
            results.append("<THIS IS Error>: Timeout Error")
        except Exception as e:
            results.append(f"<THIS IS Error>: Execution Error: {str(e)}")
        finally:
            if os.path.exists(tmp_script_path):
                os.remove(tmp_script_path)

    return results

def run_scripts_with_timeout_with_usuage(script, inputs, time_limit):
    import psutil
    results = []

    for input_val in inputs:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_script_file:
                tmp_script_file.write(script)
                tmp_script_path = tmp_script_file.name

            result = subprocess.run(
                ["python3", tmp_script_path],
                input=input_val,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=time_limit,
            )

            if result.returncode == 0:
                results.append(result.stdout.strip())
            else:
                results.append(f"<THIS IS Error>: RuntimeError: {result.stderr.strip()}")

        except subprocess.TimeoutExpired:
            results.append("<THIS IS Error>: Timeout Error")
        except Exception as e:
            results.append(f"<THIS IS Error>: Execution Error: {str(e)}")
        finally:
            if os.path.exists(tmp_script_path):
                os.remove(tmp_script_path)

    return results

