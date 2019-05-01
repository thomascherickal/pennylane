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
"""QGT optimizer"""
import autograd
import autograd.numpy as np

from pennylane.utils import _flatten, unflatten

from .gradient_descent import GradientDescentOptimizer


class QGTOptimizer(GradientDescentOptimizer):
    r"""Optimizer with adaptive learning rate.

    .. note::

        This optimizer **only supports single QNodes** as objective functions.

    Args:
        stepsize (float): the user-defined stepsize parameter :math:`\eta`
        tol (float): tolerance used for inverse
    """
    def __init__(self, stepsize=0.01, tol=1e-6):
        super().__init__(stepsize)
        self.metric_tensor = None
        self.tol = tol

    def compute_grad(self, objective_fn, x, grad_fn=None):
        r"""Compute gradient of the QNode at the point x.

        Args:
            objective_fn (QNode): the QNode for optimization
            x (array): NumPy array containing the current values of the variables to be updated

        Returns:
            array: NumPy array containing the gradient :math:`\nabla f(x^{(t)})`
        """
        if grad_fn is not None:
            # Maybe this should be a warning instead, rather than an exception
            raise ValueError("QGTOptimizer must not be passed a gradient function.")

        if self.metric_tensor is None:
            # if the metric tensor has not been calculated,
            # first we must construct the subcircuits before
            # we call the gradient function
            try:
                # Note: we pass the parameters 'x' to this method,
                # but the values themselves are not used.
                # Rather, they are simply needed for the JIT
                # circuit construction, to determine expected parameter shapes.
                objective_fn.construct_subcircuits([x])
            except AttributeError:
                raise ValueError("The objective_fn must be a QNode.")

        # calling the gradient function will implicitly
        # evaluate the subcircuit expectations
        g = autograd.grad(objective_fn)(x)  # pylint: disable=no-value-for-parameter

        if self.metric_tensor is None:
            # calculate metric tensor elements
            self.metric_tensor = np.zeros_like(x.flatten())

            for i in range(len(x.flatten())):
                # evaluate metric tensor diagonals
                # Negative occurs due to generator convention
                expval = -objective_fn.subcircuits[i]['result']

                # calculate variance
                self.metric_tensor[i] = 0.5 * expval - 0.25 * expval ** 2

        return g

    def apply_grad(self, grad, x):
        r"""Update the variables x to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (array): The gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            x (array): the current value of the variables :math:`x^{(t)}`

        Returns:
            array: the new values :math:`x^{(t+1)}`
        """
        grad_flat = list(_flatten(grad))
        x_flat = _flatten(x)

        # inverse metric tensor
        G_inv = np.where(np.abs(self.metric_tensor) > self.tol, 1/self.metric_tensor, 0)

        x_new_flat = [e - self._stepsize * g * d for e, g, d in zip(x_flat, G_inv, grad_flat)]

        return unflatten(x_new_flat, x)
