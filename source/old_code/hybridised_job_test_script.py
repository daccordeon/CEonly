#!/usr/bin/env python3
"""James Gardner, April 2022"""
import sys, os
from useful_functions import parallel_map
import time

sys_args = sys.argv[1:]

if len(sys_args) != 0:
    task_id, num_cpus, args = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
else:
    task_id, num_cpus = 0, 4
print(f"system arguments to task {task_id}: {sys_args}")


def sqr(x):
    return x**2


def fn(x):
    time.sleep(0.1)
    return sqr


ins = list(range(10))
t0 = time.time()
# fn cannot contain inner functions or calls to lambdas
outs = parallel_map(fn, ins, parallel=True, unordered=True, num_cpus=num_cpus)
t1 = time.time()
print(
    f"task {task_id}, parallel calculation took {t1 - t0} on {num_cpus}/{os.cpu_count()} available cores"
)
