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
Default parameters, commandline arguments and common routines for the unit tests.
"""
import pytest
import numpy as np

import openqml as qm

# defaults
TOLERANCE = 1e-5
HBAR = 2


# command line arguments
def cmd_args(parser):
    parser.addoption('-t', '--tol', type=float, help='Numerical tolerance for equality tests.')
    parser.addoption('--hbar', type=float, help='Hbar value.')


@pytest.fixture
def tol(request):
    try:
        return request.config.getoption("--tol")
    except:
        return TOLERANCE

@pytest.fixture
def hbar(request):
    try:
        return request.config.getoption("--hbar")
    except:
        return HBAR


@pytest.fixture
def qubit_dev1():
    return qm.device('default.qubit', wires=1)


@pytest.fixture
def qubit_dev2():
    return qm.device('default.qubit', wires=2)


@pytest.fixture
def fock_dev1(hbar):
    return qm.device('strawberryfields.fock', wires=1, hbar=hbar, cutoff_dim=25)


@pytest.fixture
def fock_dev2(hbar):
    return qm.device('strawberryfields.fock', wires=2, hbar=hbar, cutoff_dim=10)


@pytest.fixture
def gaussian_dev1():
    return qm.device('strawberryfields.gaussian', wires=1)


@pytest.fixture
def gaussian_dev2():
    return qm.device('strawberryfields.gaussian', wires=2)


class BaseTest:
    """Default base test class"""

    def assertEqual(self, first, second):
        """Replaces unittest TestCase.assertEqual"""
        assert first == second

    def assertAlmostEqual(self, first, second, delta):
        """Replaces unittest TestCase.assertEqual"""
        assert np.abs(first - second) <= delta

    def assertTrue(self, first):
        """Replaces unittest TestCase.assertTrue"""
        assert first

    def assertFalse(self, first):
        """Replaces unittest TestCase.assertFalse"""
        assert not first

    def assertAllAlmostEqual(self, first, second, delta, msg=None):
        """
        Like assertAlmostEqual, but works with arrays. All the corresponding elements have to be almost equal.
        """
        if isinstance(first, tuple):
            # check each element of the tuple separately (needed for when the tuple elements are themselves batches)
            if np.all([np.all(first[idx] == second[idx]) for idx, _ in enumerate(first)]):
                return
            if np.all([np.all(np.abs(first[idx] - second[idx])) <= delta for idx, _ in enumerate(first)]):
                return
        else:
            if np.all(first == second):
                return
            if np.all(np.abs(first - second) <= delta):
                return
        assert False, '{} != {} within {} delta'.format(first, second, delta)

    def assertAllEqual(self, first, second, msg=None):
        """
        Like assertEqual, but works with arrays. All the corresponding elements have to be equal.
        """
        return self.assertAllAlmostEqual(first, second, delta=0.0, msg=msg)

    def assertAllTrue(self, value, msg=None):
        """
        Like assertTrue, but works with arrays. All the corresponding elements have to be True.
        """
        return self.assertTrue(np.all(value))

    def assertAlmostLess(self, first, second, delta, msg=None):
        """
        Like assertLess, but with a tolerance.
        """
        return self.assertLess(first, second+delta, msg=msg)
