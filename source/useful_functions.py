"""Miscellaneous useful mathematical and python functions that are not implementation specific.

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
from typing import List, Set, Dict, Tuple, Optional, Union, Type, Callable, Iterable
from types import TracebackType
from numpy.typing import NDArray
import os, sys
import numpy as np
from p_tqdm import p_map, p_umap
from multiprocessing import Pool


class HiddenPrints(object):
    """Class used in with statements to hide stdout, e.g. print statements.

    From <https://stackoverflow.com/a/45669280>; use as ``with HiddenPrints():''.
    Typing from <https://stackoverflow.com/questions/49959656/typing-exit-in-3-5-fails-on-runtime-but-typechecks>, how does this work for exc_value without Type?
    """

    # TODO: Add docstrings to these private methods once I'm familiar with them, see above urls for now.
    def __enter__(self) -> None:
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        sys.stdout.close()
        sys.stdout = self._original_stdout


class PassEnterExit(object):
    """Class used in with statements to just execute the block of code.

    Used to replace HiddenPrints in with statements to allow prints.
    """

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        pass


def without_rows_w_nan(xarr: NDArray) -> NDArray:
    """Returns an array with all rows (2nd axis) containing NaNs filtered out.

    From <https://note.nkmk.me/en/python-numpy-nan-remove/>.

    Args:
        xarr: Array to filter.
    """
    return xarr[np.logical_not(np.isnan(xarr).any(axis=1))]


def sigmoid_3parameter(z: float, a: float, b: float, c: float) -> float:
    """Returns the modified (3-parameter) sigmoid function evaluated at a point.

    For c = 1 it is the regular sigmoid.

    Args:
        z: Point to evaluate at.
        a: First sigmoid parameter.
        b: Second sigmoid parameter.
        c: Third, now free, sigmoid parameter.
    """
    return ((1 + b) / (1 + b * np.exp(a * z))) ** c


def flatten_list(x: list[list]) -> list:
    """Returns a list with two levels flattened, e.g. flattens a 2D list.

    TODO: remove bug that it affects strings, e.g. x = ['ab', 'cd'].

    Args:
        x: 2D list to flatten.
    """
    # multiple list comprehension evaluated left-to-right like nested for loops but with the last code block as the initial item
    return [z for y in x for z in y]


def parallel_map(
    fn: Callable[[Any], Any],
    xarr: Iterable,
    display_progress_bar: bool = False,
    unordered: bool = False,
    num_cpus: int = os.cpu_count(),
    parallel: bool = True,
) -> list:
    """Returns a function mapped over an iterable using parallel computation.

    Direct substitution doesn't work because pool.map and p_map work differently, e.g. the latter can take fn as a lambda itself (but still no internal functions or lambda calls).

    Args:
        fn: Function to map from elements of xarr. Cannot be pickled if it contains inner functions or calls to lambdas, TODO: change to dill to allow this?
        xarr: Elements to map over.
        display_progress_bar: Whether to use tqdm to display a progress bar.
        unordered: Whether to let the parallel computations occur out of order.
        num_cpus: The number of CPUs to use.
        parallel: Whether to parallelize the computation.
    """
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


def logarithmically_uniform_sample(
    low: float, high: float, num_samples: int, seed: Optional[int] = None
) -> NDArray[float]:
    """Returns a number of samples uniformly distributed when viewed on a logarithmic scale.

    Done by uniformly sampling the log-transform variable.
    From <https://stackoverflow.com/a/43977980>.

    Args:
        low: Minimum value.
        high: Maximum value.
        num_samples: Number of samples to draw.
        seed: Random seed to generate samples.
    """
    # TODO: update seeding to the best practice described here: https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
    return np.exp(
        np.random.default_rng(seed).uniform(
            low=np.log(low), high=np.log(high), size=num_samples
        )
    )


def insert_at_pattern(initial: str, insert: str, pattern: str) -> str:
    """Returns a string with a different string inserted at the first matching pattern.

    Args:
        initial: String to search over.
        insert: String to replace with.
        pattern: String to replace the first instance of.
    """
    insert_index = initial.find(pattern)
    return initial[:insert_index] + insert + initial[insert_index:]
