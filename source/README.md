### CEonlyPony/source/
*Source code for processing and plotting*

James Gardner, April 2022

Current build found [here](https://github.com/daccordeon/CEonlyPony).

---
- Contact for any technical enquiries is <james.gardner@anu.edu.au>.
- See the main README.md for general notes and requirements.

---
The critical path to generate injections and create detection rate and measurement error CDF plots is as follows:
- Execute sbatch job_scripts/job_run_all_networks_injections_at_scale.sh (job_scripts/job_run_all_networks_injections_at_scale_on_sstar.sh) on OzStar's farnarkle 1/2 (sstar) login node
    - This calls run_injections_for_network_id.py over each combinatation of network and science case (with a unique waveform per science case) for a number of primitively parallel tasks
    - This calls calculate_injections.py which initialises the network, generates the injections, and then benchmarks them by calling basic_benchmarking.py which uses gwbench. If symbolic waveforms are used, then their derivatives should first be generated and saved into lambdified_functions by executing python3 run_generate_symbolic_derivatives_for_all_locations.py. The results from each primitively parallel task are saved as .npy files in data_redshift_snr_errs_sky-area.
- Execute python3 merge_npy_files.py to combine the task files into one .npy data file for each combination of network and science case.
- During the injections, the following modules are used:
    - constants.py to define physical constants, known/measured values, and thresholds
    - filename_search_and_manipulation.py to manipulate the labels of the networks and data files
    - merger_and_detection_rates.py to define the cosmological merger rates
    - networks.py to list all studied networks
    - network_subclass.py to add commonly requested attributes to gwbench's Network class
    - useful_functions.py to define some commonly used functions, e.g. the 3-parameter sigmoid
- To then generate all of the plots into the plots directory, execute sbatch job_plot_detection_rate_and_measurement_errors.sh which calls plot_collated_detection_rate.py and plot_collated_PDFs_and_CDFs.py from the .npy data files loaded in a results class instance defined in  results_class.py.
- Asides from the above modules, the following are additionally used:
    - cosmological_redshift_resampler.py to resample the linearly uniformly sampled redshifts using a cosmological model
    - useful_plotting_functions.py to define some commonly used plotting functions
- All other files are non-critical and used for testing, e.g. profiling the memory usage of other processes

---
file structure
```bash
.
├── data_redshift_snr_errs_sky-area
│         └── ...
├── gwbench
│         └── ...
├── job_scripts
│         ├── job_memory_profile_all_science-cases_injections.sh
│         ├── job_memory_profile_plot_detection_rate_and_measurement_errors.sh
│         ├── job_plot_detection_rate_and_measurement_errors.sh
│         ├── job_run_all_networks_injections_at_scale_on_sstar.sh
│         └── job_run_all_networks_injections_at_scale.sh
├── lambdified_functions
│         └── ...
├── old_code
│         └── ...
├── plots
│         └── ...
├── basic_benchmarking.py
├── calculate_injections.py
├── constants.py
├── cosmological_redshift_resampler.py
├── filename_search_and_manipulation.py
├── merge_npy_files.py
├── merger_and_detection_rates.py
├── networks.py
├── network_subclass.py
├── plot_collated_detection_rate.py
├── plot_collated_PDFs_and_CDFs.py
├── results_class.py
├── run_generate_symbolic_derivatives_for_all_locations.py
├── run_injections_for_hard-coded_network.py
├── run_injections_for_network_id.py
├── run_plot_detection_rate_and_measurement_errors.py
├── useful_functions.py
├── useful_plotting_functions.py
├── workshop.ipynb
├── jn_remote_join.sh
└── README.md
```
[//]: # (tree -I '*.pdf|*.png')