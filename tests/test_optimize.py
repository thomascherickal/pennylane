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
Unit tests for the :mod:`openqml` :class:`QNode` class.
"""
import pytest
from conftest import BaseTest

import openqml as qm

from openqml import numpy as np
from openqml.optimize import (GradientDescentOptimizer,
                              MomentumOptimizer,
                              NesterovMomentumOptimizer,
                              AdagradOptimizer,
                              RMSPropOptimizer,
                              AdamOptimizer)

x_vals = np.linspace(-10, 10, 16, endpoint=False)

# Hyperparameters for optimizers
stepsize = 0.1
gamma = 0.5
delta = 0.8

univariate_funcs = [np.sin,
                         lambda x: np.exp(x / 10.),
                         lambda x: x ** 2]
grad_uni_fns = [np.cos,
                     lambda x: np.exp(x / 10.) / 10.,
                     lambda x: 2 * x]
multivariate_funcs = [lambda x: np.sin(x[0]) + np.cos(x[1]),
                           lambda x: np.exp(x[0] / 3) * np.tanh(x[1]),
                           lambda x: np.sum(x_ ** 2 for x_ in x)]
grad_multi_funcs = [lambda x: np.array([np.cos(x[0]), -np.sin(x[1])]),
                         lambda x: np.array([np.exp(x[0] / 3) / 3 * np.tanh(x[1]),
                                        np.exp(x[0] / 3) * (1 - np.tanh(x[1]) ** 2)]),
                         lambda x: np.array([2 * x_ for x_ in x])]



sgd_opt = GradientDescentOptimizer(stepsize)
mom_opt = MomentumOptimizer(stepsize, momentum=gamma)
nesmom_opt = NesterovMomentumOptimizer(stepsize, momentum=gamma)
adag_opt = AdagradOptimizer(stepsize)
rms_opt = RMSPropOptimizer(stepsize, decay=gamma)
adam_opt = AdamOptimizer(stepsize, beta1=gamma, beta2=delta)


class TestBasics(BaseTest):
    """Basic optimizer tests.
    """
    def test_gradient_descent_optimizer(self, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly."""

        for gradf, f in zip(grad_uni_fns, univariate_funcs):
            for x_start in x_vals:
                x_new = sgd_opt.step(f, x_start)
                x_correct = x_start - gradf(x_start) * stepsize
                self.assertAlmostEqual(x_new, x_correct, delta=tol)

        for gradf, f in zip(grad_multi_funcs, multivariate_funcs):
            for jdx in range(len(x_vals[:-1])):
                x_vec = x_vals[jdx:jdx+2]
                x_new = sgd_opt.step(f, x_vec)
                x_correct = x_vec - gradf(x_vec) * stepsize
                self.assertAllAlmostEqual(x_new, x_correct, delta=tol)

    def test_momentum_optimizer(self, tol):
        """Tests that momentum optimizer takes one and two steps correctly."""

        for gradf, f in zip(grad_uni_fns, univariate_funcs):
            for x_start in x_vals:
                mom_opt.reset()  # TODO: Is it better to recreate the optimizer?

                x_onestep = mom_opt.step(f, x_start)
                x_onestep_target = x_start - gradf(x_start) * stepsize
                self.assertAlmostEqual(x_onestep, x_onestep_target, delta=tol)

                x_twosteps = mom_opt.step(f, x_onestep)
                momentum_term = gamma * gradf(x_start)
                x_twosteps_target = x_onestep - (gradf(x_onestep) + momentum_term) * stepsize
                self.assertAlmostEqual(x_twosteps, x_twosteps_target, delta=tol)

        for gradf, f in zip(grad_multi_funcs, multivariate_funcs):
            for jdx in range(len(x_vals[:-1])):
                mom_opt.reset()

                x_vec = x_vals[jdx:jdx + 2]
                x_onestep = mom_opt.step(f, x_vec)
                x_onestep_target = x_vec - gradf(x_vec) * stepsize
                self.assertAllAlmostEqual(x_onestep, x_onestep_target, delta=tol)

                x_twosteps = mom_opt.step(f, x_onestep)
                momentum_term = gamma * gradf(x_vec)
                x_twosteps_target = x_onestep - (gradf(x_onestep) + momentum_term) * stepsize
                self.assertAllAlmostEqual(x_twosteps, x_twosteps_target, delta=tol)

    def test_nesterovmomentum_optimizer(self, tol):
        """Tests that nesterov momentum optimizer takes one and two steps correctly."""

        for gradf, f in zip(grad_uni_fns, univariate_funcs):
            for x_start in x_vals:
                nesmom_opt.reset()

                x_onestep = nesmom_opt.step(f, x_start)
                x_onestep_target = x_start - gradf(x_start) * stepsize
                self.assertAlmostEqual(x_onestep, x_onestep_target, delta=tol)

                x_twosteps = nesmom_opt.step(f, x_onestep)
                momentum_term = gamma * gradf(x_start)
                shifted_grad_term = gradf(x_onestep - stepsize * momentum_term)
                x_twosteps_target = x_onestep - (shifted_grad_term + momentum_term) * stepsize
                self.assertAlmostEqual(x_twosteps, x_twosteps_target, delta=tol)

        for gradf, f in zip(grad_multi_funcs, multivariate_funcs):
            for jdx in range(len(x_vals[:-1])):
                nesmom_opt.reset()

                x_vec = x_vals[jdx:jdx + 2]
                x_onestep = nesmom_opt.step(f, x_vec)
                x_onestep_target = x_vec - gradf(x_vec) * stepsize
                self.assertAllAlmostEqual(x_onestep, x_onestep_target, delta=tol)

                x_twosteps = nesmom_opt.step(f, x_onestep)
                momentum_term = gamma * gradf(x_vec)
                shifted_grad_term = gradf(x_onestep - stepsize * momentum_term)
                x_twosteps_target = x_onestep - (shifted_grad_term + momentum_term) * stepsize
                self.assertAllAlmostEqual(x_twosteps, x_twosteps_target, delta=tol)

    def test_adagrad_optimizer(self, tol):
        """Tests that adagrad optimizer takes one and two steps correctly."""

        for gradf, f in zip(grad_uni_fns, univariate_funcs):
            for x_start in x_vals:
                adag_opt.reset()

                x_onestep = adag_opt.step(f, x_start)
                past_grads = gradf(x_start)*gradf(x_start)
                adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                x_onestep_target = x_start - gradf(x_start) * adapt_stepsize
                self.assertAlmostEqual(x_onestep, x_onestep_target, delta=tol)

                x_twosteps = adag_opt.step(f, x_onestep)
                past_grads = gradf(x_start)*gradf(x_start) + gradf(x_onestep)*gradf(x_onestep)
                adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
                self.assertAlmostEqual(x_twosteps, x_twosteps_target, delta=tol)

        for gradf, f in zip(grad_multi_funcs, multivariate_funcs):
            for jdx in range(len(x_vals[:-1])):
                adag_opt.reset()

                x_vec = x_vals[jdx:jdx + 2]
                x_onestep = adag_opt.step(f, x_vec)
                past_grads = gradf(x_vec)*gradf(x_vec)
                adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                x_onestep_target = x_vec - gradf(x_vec) * adapt_stepsize
                self.assertAllAlmostEqual(x_onestep, x_onestep_target, delta=tol)

                x_twosteps = adag_opt.step(f, x_onestep)
                past_grads = gradf(x_vec) * gradf(x_vec) + gradf(x_onestep) * gradf(x_onestep)
                adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
                x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
                self.assertAllAlmostEqual(x_twosteps, x_twosteps_target, delta=tol)

    def test_rmsprop_optimizer(self, tol):
        """Tests that rmsprop optimizer takes one and two steps correctly."""

        for gradf, f in zip(grad_uni_fns, univariate_funcs):
            for x_start in x_vals:
                rms_opt.reset()

                x_onestep = rms_opt.step(f, x_start)
                past_grads = (1 - gamma) * gradf(x_start)*gradf(x_start)
                adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                x_onestep_target = x_start - gradf(x_start) * adapt_stepsize
                self.assertAlmostEqual(x_onestep, x_onestep_target, delta=tol)

                x_twosteps = rms_opt.step(f, x_onestep)
                past_grads = (1 - gamma) * gamma * gradf(x_start)*gradf(x_start) \
                             + (1 - gamma) * gradf(x_onestep)*gradf(x_onestep)
                adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
                self.assertAlmostEqual(x_twosteps, x_twosteps_target, delta=tol)

        for gradf, f in zip(grad_multi_funcs, multivariate_funcs):
            for jdx in range(len(x_vals[:-1])):
                rms_opt.reset()

                x_vec = x_vals[jdx:jdx + 2]
                x_onestep = rms_opt.step(f, x_vec)
                past_grads = (1 - gamma) * gradf(x_vec)*gradf(x_vec)
                adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                x_onestep_target = x_vec - gradf(x_vec) * adapt_stepsize
                self.assertAllAlmostEqual(x_onestep, x_onestep_target, delta=tol)

                x_twosteps = rms_opt.step(f, x_onestep)
                past_grads = (1 - gamma) * gamma * gradf(x_vec) * gradf(x_vec) \
                             + (1 - gamma) * gradf(x_onestep) * gradf(x_onestep)
                adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
                x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
                self.assertAllAlmostEqual(x_twosteps, x_twosteps_target, delta=tol)

    def test_adam_optimizer(self, tol):
        """Tests that adam optimizer takes one and two steps correctly."""

        for gradf, f in zip(grad_uni_fns, univariate_funcs):
            for x_start in x_vals:
                adam_opt.reset()

                x_onestep = adam_opt.step(f, x_start)
                adapted_stepsize = stepsize * np.sqrt(1 - delta)/(1 - gamma)
                firstmoment = gradf(x_start)
                secondmoment = gradf(x_start) * gradf(x_start)
                x_onestep_target = x_start - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
                self.assertAlmostEqual(x_onestep, x_onestep_target, delta=tol)

                x_twosteps = adam_opt.step(f, x_onestep)
                adapted_stepsize = stepsize * np.sqrt(1 - delta**2) / (1 - gamma**2)
                firstmoment = (gamma * gradf(x_start) + (1 - gamma) * gradf(x_onestep))
                secondmoment = (delta * gradf(x_start) * gradf(x_start) + (1 - delta) * gradf(x_onestep) * gradf(x_onestep))
                x_twosteps_target = x_onestep - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
                self.assertAlmostEqual(x_twosteps, x_twosteps_target, delta=tol)

        for gradf, f in zip(grad_multi_funcs, multivariate_funcs):
            for jdx in range(len(x_vals[:-1])):
                adam_opt.reset()

                x_vec = x_vals[jdx:jdx + 2]
                x_onestep = adam_opt.step(f, x_vec)
                adapted_stepsize = stepsize * np.sqrt(1 - delta) / (1 - gamma)
                firstmoment = gradf(x_vec)
                secondmoment = gradf(x_vec) * gradf(x_vec)
                x_onestep_target = x_vec - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
                self.assertAllAlmostEqual(x_onestep, x_onestep_target, delta=tol)

                x_twosteps = adam_opt.step(f, x_onestep)
                adapted_stepsize = stepsize * np.sqrt(1 - delta**2) / (1 - gamma**2)
                firstmoment = (gamma * gradf(x_vec) + (1 - gamma) * gradf(x_onestep))
                secondmoment = (delta * gradf(x_vec) * gradf(x_vec) + (1 - delta) * gradf(x_onestep) * gradf(x_onestep))
                x_twosteps_target = x_onestep - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
                self.assertAllAlmostEqual(x_twosteps, x_twosteps_target, delta=tol)
