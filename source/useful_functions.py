"""James Gardner, March 2022"""
import os, sys
import numpy as np
from p_tqdm import p_map, p_umap
from multiprocessing import Pool

class PassEnterExit:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# https://stackoverflow.com/a/45669280; use as ``with HiddenPrints():''
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# https://note.nkmk.me/en/python-numpy-nan-remove/
def without_rows_w_nan(xarr): return xarr[np.logical_not(np.isnan(xarr).any(axis=1))]

# the modified sigmoid function with c free, for c=1 it is the regular sigmoid
def sigmoid_3parameter(z, a, b, c): return ((1+b)/(1+b*np.exp(a*z)))**c

# x = [y, ...], y = [z, ...]
def flatten_list(x): return [z for y in x for z in y]

def parallel_map(f, x, display_progress_bar=False, unordered=False, num_cpus=os.cpu_count()):
    """f is a function to apply to elements in iterable x,
    display_progress_bar is a bool about whether to use tqdm;
    returns a list.
    direct substitution doesn't work because pool.map and p_map work differently,
    e.g. the latter can take lambdas"""
    if display_progress_bar:
        if unordered:
            return list(p_umap(f, x, num_cpus=num_cpus))
        else:
            return list(p_map( f, x, num_cpus=num_cpus))
    else:
        global _global_copy_of_f
        def _global_copy_of_f(x0):
            return f(x0)
        
        with Pool(processes=num_cpus) as pool:
            if unordered:
                return list(pool.imap_unordered(_global_copy_of_f, x))
            else:
                return list(pool.map(_global_copy_of_f, x))  
