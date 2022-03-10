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
without_rows_w_nan = lambda xarr : xarr[np.logical_not(np.isnan(xarr).any(axis=1))]

sigmoid_3parameter = lambda z, a, b, c : ((1+b)/(1+b*np.exp(a*z)))**c

flatten_list = lambda x : [z for y in x for z in y] # x = [y, ...], y = [z, ...]

def parallel_map(f, x, display_progress_bar=False, unordered_if_possible=False, **kwargs):
    """f is a function to apply to elements in iterable x,
    display_progress_bar is a bool about whether to use tqdm;
    returns a list.
    direct substitution doesn't work because pool.map and p_map work differently,
    e.g. the latter can take lambdas"""
    if display_progress_bar:
        if unordered_if_possible:
            return list(p_umap(f, x, **kwargs))
        else:
            return list(p_map(f, x, **kwargs))
    else:
        global _global_copy_of_f
        def _global_copy_of_f(x0):
            return f(x0)
        
        with Pool() as pool:
            # to-do: add unordered using imap or imap_unordered?
            if unordered_if_possible:
                return list(pool.imap_unordered(_global_copy_of_f, x, **kwargs))
            else:
                return list(pool.map(_global_copy_of_f, x, **kwargs))  
