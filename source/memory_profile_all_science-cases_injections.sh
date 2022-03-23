#!/bin/bash
# James Gardner, March 2022

SCIENCE_CASES=('BNS' 'BBH')
#NUM_INJS_PER_ZBIN_PER_TASK_LIST=(100 10) # 10 numerical injections/zbin takes 3 minutes on a single core
NUM_INJS_PER_ZBIN_PER_TASK_LIST=(1000 1000) # want to know how long it'll take to scale up to B&S2022

for ((i=0; i<${#SCIENCE_CASES[@]}; i++ )); #{0..{${#SCIENCE_CASES[*]}-1}}
do
    # ensure that @profile is enabled in detection_rates.py and that data files don't already exist (if they do, then just increment the task ID, currently 0, below)
    #echo "mprof_plot_${SCIENCE_CASES[$i]}_${NUM_INJS_PER_ZBIN_PER_TASK_LIST[$i]}.pdf"
    mprof run old_run_injections_for_hard-coded_network.py 0 ${NUM_INJS_PER_ZBIN_PER_TASK_LIST[$i]} ${SCIENCE_CASES[$i]}; mprof plot -o "mprof_plot_${SCIENCE_CASES[$i]}_${NUM_INJS_PER_ZBIN_PER_TASK_LIST[$i]}.pdf"
done

