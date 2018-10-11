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
Unit tests for :mod:`openqml.operation`.
"""
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()

import numpy as np
import numpy.random as nr

from conftest import BaseTest

import openqml as qm
import openqml.qnode as oq
import openqml.operation as oo
import openqml.variable as ov


class TestBasic(BaseTest):
    """Utility class tests."""
    def test_heisenberg(self, tol):
        "Heisenberg picture adjoint actions of CV Operations."

        def h_test(cls):
            "Test a gaussian CV operation."
            log.debug('\tTesting: cls.__name__')
            # fixed parameter values
            if cls.par_domain == 'A':
                par = [nr.randn(1,1)] * cls.n_params
            else:
                par = list(nr.randn(cls.n_params))
            ww = list(range(cls.n_wires))
            op = cls(*par, wires=ww, do_queue=False)

            if issubclass(cls, oo.Expectation):
                Q = op.heisenberg_obs(0)
                # ev_order equals the number of dimensions of the H-rep array
                self.assertEqual(Q.ndim, cls.ev_order)
                return

            # not an Expectation
            # all gaussian ops use the 'A' method
            self.assertEqual(cls.grad_method, 'A')
            U = op.heisenberg_tr(2)
            I = np.eye(*U.shape)
            # first row is always (1,0,0...)
            self.assertAllEqual(U[0,:], I[:,0])

            # check the inverse transform
            V = op.heisenberg_tr(2, inverse=True)
            self.assertAlmostEqual(np.linalg.norm(U @ V -I), 0, delta=tol)
            self.assertAlmostEqual(np.linalg.norm(V @ U -I), 0, delta=tol)

            # compare gradient recipe to numerical gradient
            h = 1e-7
            U = op.heisenberg_tr(0)
            for k in range(cls.n_params):
                D = op.heisenberg_pd(k)  # using the recipe
                # using finite differences
                op.params[k] += h
                Up = op.heisenberg_tr(0)
                op.params = par
                G = (Up-U) / h
                self.assertAllAlmostEqual(D, G, delta=tol)

        for cls in qm.ops.builtins_continuous.all_ops + qm.expectation.builtins_continuous.all_ops:
            if cls._heisenberg_rep is not None:  # only test gaussian operations
                h_test(cls)


    def test_ops(self, tol):
        "Operation initialization."

        def op_test(cls):
            "Test the Operation subclass."
            log.debug('\tTesting: cls.__name__')
            n = cls.n_params
            w = cls.n_wires
            ww = list(range(w))
            # valid pars
            if cls.par_domain == 'A':
                pars = [np.eye(2)] * n
            elif cls.par_domain == 'N':
                pars = [0] * n
            else:
                pars = [0.0] * n

            # valid call
            cls(*pars, wires=ww, do_queue=False)

            # too many parameters
            with pytest.raises(ValueError, message='wrong number of parameters'):
                cls(*(n+1)*[0], wires=ww, do_queue=False)

            # too few parameters
            if n > 0:
                with pytest.raises(ValueError, message='wrong number of parameters'):
                    cls(*(n-1)*[0], wires=ww, do_queue=False)

            if w > 0:
                # too many or too few wires
                with pytest.raises(ValueError, message='wrong number of wires'):
                    cls(*pars, wires=list(range(w+1)), do_queue=False)
                with pytest.raises(ValueError, message='wrong number of wires'):
                    cls(*pars, wires=list(range(w-1)), do_queue=False)
                # repeated wires
                if w > 1:
                    with pytest.raises(ValueError, message='wires must be unique'):
                        cls(*pars, wires=w*[0], do_queue=False)

            if n == 0:
                return

            # wrong parameter types
            if cls.par_domain == 'A':
                # params must be arrays
                with pytest.raises(TypeError, message='Array parameter expected'):
                    cls(*n*[0.0], wires=ww, do_queue=False)
            elif cls.par_domain == 'N':
                # params must be natural numbers
                with pytest.raises(TypeError, message='Natural number'):
                    cls(*n*[0.7], wires=ww, do_queue=False)
                with pytest.raises(TypeError, message='Natural number'):
                    cls(*n*[-1], wires=ww, do_queue=False)
            else:
                # params must be real numbers
                with pytest.raises(TypeError, message='Real scalar parameter expected'):
                    cls(*n*[1j], wires=ww, do_queue=False)


        for cls in qm.ops.builtins_discrete.all_ops:
            op_test(cls)

        for cls in qm.ops.builtins_continuous.all_ops:
            op_test(cls)

        for cls in qm.expectation.builtins_discrete.all_ops:
            op_test(cls)

        for cls in qm.expectation.builtins_continuous.all_ops:
            op_test(cls)
