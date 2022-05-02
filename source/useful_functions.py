"""Short one-sentence description.

Long description.

Usage:
    Describe the typical usage.

License:
    BSD 3-Clause License

    Copyright (c) 2022, James Gardner.
    All rights reserved except for those for the gwbench code which remain reserved
    by S. Borhanian; the gwbench code is included in this repository for convenience.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from typing import List, Set, Dict, Tuple, Optional, Union
from numpy.typing import NDArray
import os, sys
import numpy as np
from p_tqdm import p_map, p_umap
from multiprocessing import Pool


class HiddenPrints:
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """

    """https://stackoverflow.com/a/45669280; use as ``with HiddenPrints():''"""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class PassEnterExit:
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """

    """do-nothing class to replace HiddenPrints with in with statements to allow prints"""

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def without_rows_w_nan(xarr):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """https://note.nkmk.me/en/python-numpy-nan-remove/"""
    return xarr[np.logical_not(np.isnan(xarr).any(axis=1))]


def sigmoid_3parameter(z, a, b, c):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """the modified sigmoid function with c free, for c=1 it is the regular sigmoid"""
    return ((1 + b) / (1 + b * np.exp(a * z))) ** c


def flatten_list(x):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """x = [y, ...], y = [z, ...]. TODO: remove bug that it affects strings"""
    return [z for y in x for z in y]


def parallel_map(
    fn,
    xarr,
    display_progress_bar=False,
    unordered=False,
    num_cpus=os.cpu_count(),
    parallel=True,
):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """fn is a function to apply to elements in iterable xarr, display_progress_bar is a bool about whether to use tqdm;
    returns a list of fn applied to xarr.
    fn cannot be pickled if it contains inner functions or calls to lambdas, TODO: change to dill to allow this?
    direct substitution doesn't work because pool.map and p_map work differently, e.g. the latter can take fn as a lambda itself (but still no internal functions or lambda calls)."""
    if parallel:
        if display_progress_bar:
            if unordered:
                return list(p_umap(fn, xarr, num_cpus=num_cpus))
            else:
                return list(p_map(fn, xarr, num_cpus=num_cpus))
        else:
            global _global_copy_of_fn

            def _global_copy_of_fn(x0):
                return fn(x0)

            with Pool(processes=num_cpus) as pool:
                if unordered:
                    return list(pool.imap_unordered(_global_copy_of_fn, xarr))
                else:
                    return list(pool.map(_global_copy_of_fn, xarr))
    else:
        return list(map(fn, xarr))


def logarithmically_uniform_sample(low, high, num_samples, seed=None):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """generates a number of samples (num_samples) in (low, high) such that they are uniformly distributed when viewed on a logarithmic scale, done by uniformly sampling the log-transform variable
    credit: https://stackoverflow.com/a/43977980"""
    # TODO: update seeding to the best practice described here: https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
    return np.exp(
        np.random.default_rng(seed).uniform(
            low=np.log(low), high=np.log(high), size=num_samples
        )
    )


def insert_at_pattern(initial, insert, pattern):
    """Short description.

    Args:
        x: _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    """given three strings, returns a copy of initial with insert inserted at the first matching pattern"""
    insert_index = initial.find(pattern)
    return initial[:insert_index] + insert + initial[insert_index:]
