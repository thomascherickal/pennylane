# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities
=========

**Module name:** :mod:`pennylane.utils`

.. currentmodule:: pennylane.utils

This module contains utilities and auxiliary functions, which are shared
across the PennyLane submodules.

.. raw:: html

    <h3>Summary</h3>

.. autosummary::
    _flatten
    _unflatten
    unflatten

.. raw:: html

    <h3>Code details</h3>
"""
import collections
import numbers

import autograd.numpy as np

from .variable  import Variable


def _flatten(x):
    """Iterate through an arbitrarily nested structure, flattening it in depth-first order.

    See also :func:`_unflatten`.

    Args:
        x (array, Iterable, other): each element of the Iterable may itself be an iterable object

    Yields:
        other: elements of x in depth-first order
    """
    if isinstance(x, np.ndarray):
        yield from _flatten(x.flat)  # should we allow object arrays? or just "yield from x.flat"?
    elif isinstance(x, collections.Iterable) and not isinstance(x, (str, bytes)):
        for item in x:
            yield from _flatten(item)
    else:
        yield x


def _unflatten(flat, model):
    """Restores an arbitrary nested structure from a flattened iterable.

    See also :func:`_flatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Returns:
        (other, array): first elements of flat arranged into the nested
        structure of model, unused elements of flat
    """
    print("_unflatten("+str(flat)+", "+str(model)+")")
    if isinstance(model, (numbers.Number, Variable)) and not isinstance(model, collections.Iterable):
        print("branch: 0")
        return flat[0], flat[1:]
    elif isinstance(model, collections.Iterable):
        print("branch 1")
        if isinstance(model, np.ndarray):
            l = model.shape[0] if model.shape != () else 0
        else:
            l = len(model)
        print("l="+str(l))
        if l == 0: #l=0 can only happen for single element np.arrays such as np.array(4) and for empty tuples
            if isinstance(model, np.ndarray):
                return np.array(flat[0], dtype=model.dtype), flat[1:]
            else:
                return (), flat

        # Now we know that model is not just iterable, but has length at least 1 and can actually be indexed.

        # For np.arrays there is a special shortcut if they are not yagged
        # We try this after some sanity checks and hope for the best
        if isinstance(model, np.ndarray):
            size = model.size
            #if size == np.prod(model.shape) and np.all([isinstance(x, np.ndarray) and x.shape == model[0].shape for x in model]):
            if size == np.prod(model.shape) and np.all([isinstance(x, np.ndarray) and x.shape == model[0].shape for x in model]):
                return np.array(flat)[:size].reshape(model.shape), flat[size:]

        # As we have to unflatten depth-first, we need to keep track of the "tail"
        # of not yet used elements but also want to yield from recursive calls of
        # _unflatten(). The easies way to do this is via a class that keeps track
        # of the "global" state of the tail while yielding:
        class Unflattener:
            def __init__(self, flat, model):
                self.flat = flat
                self.model = model
                self.encountered_variable = False

            def gen(self):
                for i, x in enumerate(model):
                    res, self.flat = _unflatten(self.flat, x)
                    if isinstance(res, Variable):
                        self.encountered_variable = True
                    yield res

            def tail(self):
                print("self.flat="+str(self.flat))
                return self.flat

            def dtype(self):
                if self.encountered_variable:
                    return object
                else:
                    return self.model.dtype

        unflattener = Unflattener(flat, model)

        if isinstance(model, np.ndarray):
            #return np.array(list(unflattener.gen()), dtype=unflattener.dtype()), unflattener.tail()
            return np.fromiter(unflattener.gen(), dtype=unflattener.dtype()), unflattener.tail()
        else:
            return (type(model))(unflattener.gen()), unflattener.tail()
    else:
        raise TypeError('Unsupported type in the model: {}'.format(type(model)))


def _unflatten_old(flat, model):
    """Restores an arbitrary nested structure from a flattened iterable.

    See also :func:`_flatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Returns:
        (other, array): first elements of flat arranged into the nested
        structure of model, unused elements of flat
    """
    if isinstance(model, np.ndarray):
        idx = model.size
        res = np.array(flat)[:idx].reshape(model.shape)
        return res, flat[idx:]
    elif isinstance(model, collections.Iterable):
        res = []
        for x in model:
            val, flat = _unflatten(flat, x)
            res.append(val)
        return res, flat
    elif isinstance(model, (numbers.Number, Variable)):
        return flat[0], flat[1:]
    else:
        raise TypeError('Unsupported type in the model: {}'.format(type(model)))


def unflatten(flat, model):
    """Wrapper for :func:`_unflatten`.
    """
    # pylint:disable=len-as-condition
    res, tail = _unflatten(np.asarray(flat), model)
    if len(tail) != 0:
        raise ValueError('Flattened iterable has more elements than the model. tail='+str(tail)+' model='+str(model))
    return res
