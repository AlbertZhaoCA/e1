# execute.py
import io
import sys
import multiprocessing
import time

sys.setrecursionlimit(100)

def worker(script, input_val, output_queue):
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
        output_queue.put(stdout_capture.getvalue())
    except SystemExit:
        output_queue.put(stdout_capture.getvalue())
    except Exception as e:
        output_queue.put(f"error: {e}")
    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin


def run_scripts_with_timeout(script, inputs, time_limit, worker):
    results = [None] * len(inputs)
    processes = []
    queues = []

    for i, input_val in enumerate(inputs):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=worker, args=(script, input_val, q))
        p.start()
        processes.append(p)
        queues.append(q)

    deadlines = [time.time() + time_limit] * len(inputs)

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
