#!/bin/bash

NUM_INJS_PER_ZBIN_PER_TASK=1000
mprof run run_injections.py 0 $NUM_INJS_PER_ZBIN_PER_TASK; mprof plot -o "mprof_plot_${NUM_INJS_PER_ZBIN_PER_TASK}.pdf"

