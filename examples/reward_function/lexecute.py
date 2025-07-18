# import subprocess
# import tempfile
# import os
# import time
# import re

# def run_scripts_with_timeout_with_usage(script, inputs, time_limit):
#     results = []

#     for input_val in inputs:
#         tmp_script_path = None

#         try:
#             # 写入临时 Python 文件
#             with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_script_file:
#                 tmp_script_file.write(script)
#                 tmp_script_path = tmp_script_file.name

#             # 使用 /usr/bin/time 获取峰值内存（单位 KB）
#             cmd = ["/usr/bin/time", "-v", "python3", tmp_script_path]

#             start_time = time.time()
#             result = subprocess.run(
#                 cmd,
#                 input=input_val,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#                 timeout=time_limit
#             )
#             end_time = time.time()
#             runtime = end_time - start_time

#             # 提取内存（Maximum resident set size）
#             match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", result.stderr)
#             memory_kb = int(match.group(1)) if match else 0
#             memory_mb = memory_kb / 1024.0

#             if result.returncode == 0:
#                 results.append({
#                     "output": result.stdout.strip(),
#                     "time": runtime,
#                     "memory": memory_mb
#                 })
#             else:
#                 results.append({
#                     "output": f"RuntimeError: {result.stderr.strip()}",
#                     "time": runtime,
#                     "memory": memory_mb
#                 })

#         except subprocess.TimeoutExpired:
#             results.append({
#                 "output": "Timeout Error",
#                 "time": time_limit,
#                 "memory": 0.0
#             })
#         except Exception as e:
#             results.append({
#                 "output": f"Execution Error: {str(e)}",
#                 "time": 0.0,
#                 "memory": 0.0
#             })
#         finally:
#             if tmp_script_path and os.path.exists(tmp_script_path):
#                 os.remove(tmp_script_path)

#     return results

# def run_scripts_with_timeout(script, inputs, time_limit):
#     results = []

#     for input_val in inputs:
#         try:
#             with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_script_file:
#                 tmp_script_file.write(script)
#                 tmp_script_path = tmp_script_file.name

#             result = subprocess.run(
#                 ["python3", tmp_script_path],
#                 input=input_val,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#                 timeout=time_limit,
#             )

#             if result.returncode == 0:
#                 results.append(result.stdout.strip())
#             else:
#                 results.append(f"<THIS IS Error>: {result.stderr.strip()}")

#         except subprocess.TimeoutExpired:
#             results.append("<THIS IS Error>: Timeout Error")
#         except Exception as e:
#             results.append(f"<THIS IS Error>: {str(e)}")
#         finally:
#             if os.path.exists(tmp_script_path):
#                 os.remove(tmp_script_path)

#     return results

import subprocess
import tempfile
import os
import py_compile
import resource
from typing import List

def run_scripts_with_timeout(
    script: str,
    inputs: List[str],
    time_limit: float = 2.5,
    *,
    interpreter: str = "python3",
    precompile: bool = True,
    keep_file: bool = False,
    max_memory_mb: float | None = None,   # 可选：对单次运行设内存上限
) -> List[str]:
    """
    执行一段 Python 代码并收集多组输入的输出结果。

    Parameters
    ----------
    script : str
        要执行的完整 Python 源代码。
    inputs : List[str]
        每次执行写入 stdin 的字符串列表。
    time_limit : float, optional
        单次执行的最大运行时间（秒）。
    interpreter : str, optional
        解释器命令，可设为 "pypy3" 加速。
    precompile : bool, optional
        是否先做语法编译检查，异常立即返回。
    keep_file : bool, optional
        True 表示保留临时脚本文件（便于 debug）。
    max_memory_mb : float | None, optional
        若给定，对子进程设置最大常驻内存 (RSS) 预算。

    Returns
    -------
    List[str]
        outputs 或错误字符串，与 `inputs` 一一对应。
    """
    results: List[str] = []
    tmp_script_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_script_file:
            tmp_script_file.write(script)
            tmp_script_path = tmp_script_file.name

        # 2️⃣ 语法预检查，提前失败快
        if precompile:
            try:
                py_compile.compile(tmp_script_path, doraise=True)
            except py_compile.PyCompileError as e:
                # 统一错误格式，直接对所有输入返回同一个语法错误
                syntax_msg = f"<THIS IS Error>: SyntaxError: {e.msg}"
                return [syntax_msg for _ in inputs]

        for input_val in inputs:
            try:
                # 可选：对子进程设置内存上限
                preexec_fn = None
                if max_memory_mb is not None:
                    def set_rlimit():
                        bytes_limit = int(max_memory_mb * 1024 * 1024)
                        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
                    preexec_fn = set_rlimit

                completed = subprocess.run(
                    [interpreter, tmp_script_path],
                    input=input_val,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=time_limit,
                    preexec_fn=preexec_fn,
                )

                if completed.returncode == 0:
                    results.append(completed.stdout.rstrip("\n"))
                else:
                    results.append(f"<THIS IS Error>: {completed.stderr.strip()}")

            except subprocess.TimeoutExpired:
                results.append("<THIS IS Error>: Timeout Error")
            except Exception as exc:
                results.append(f"<THIS IS Error>: {str(exc)}")

    finally:
        if tmp_script_path and not keep_file and os.path.exists(tmp_script_path):
            os.remove(tmp_script_path)

    return results

