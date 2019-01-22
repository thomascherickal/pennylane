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
Unit tests for the :mod:`pennylane.utils` sub-module.
"""
import types
import unittest
import logging as log
log.getLogger('defaults')

import autograd.numpy as np

import collections

from defaults import pennylane as qml, BaseTest

from pennylane.utils import _flatten as flatten, unflatten

a = np.linspace(-1, 1, 64)
a_shapes = [(64,),
            (64, 1),
            (32, 2),
            (16, 4),
            (8, 8),
            (16, 2, 2),
            (8, 2, 2, 2),
            (4, 2, 2, 2, 2),
            (2, 2, 2, 2, 2, 2)]

b = np.linspace(-1., 1., 8)

#flattened and non-flattened version of various array - simple and crazy ones
crazy_arrays = [
    ([0], [0]),
    ([0, 1], [0, 1]),
    ([0, 1, 2], [0, 1, 2]),
    ([0, 1], [0, [1]]),
    ([0, 1, 2], [0, [1, 2]]),
    (np.array([0, 1, 2]), [0, [1, 2]]),
    ([0, 1, 2], np.array([0, 1, 2])),
    ([0, 1, 2], np.array([[0, 1], 2])),
    ([0, 1, 2], np.array([0, [1, 2]], dtype=object)),
    ([1,2], [1,2]),
    ([1,2], (1,2)),
    ([1,2,3], (1,(2,3))),
    ([1,2,3], ((1,2),3)),
    ([1,2,3,4,5], (((1,[2,3]),4),5)),
    (np.array([0, 1, 2]), np.array([[0, 1], 2])),
    (np.array([0, 1, 2]), np.array([0, [1, 2]], dtype=object)),
    (np.array([0, 1, 2, 3]), np.array([[0, 1], [2, 3]])),
    ([0, 1, 2, 3, 4], [[0, 1], [(2, 3), 4]]),
    ([0, 1, 2, 3, 4], [[0, 1], [(2, 3), 4]]),
    (np.array(range(17)), [np.array([np.array([0]), np.array([1, 2, 3]), np.array([4, 5])]), 6, np.array([7, 8]), (9, 10), [11, (12, np.array(13)), np.array([(14, ), 15, np.array(16)])]]),
    ([0, 1, 2], np.array([np.array([0, 1], dtype=object), 2])),
    #(np.array([0, 1, 2, 3, 4]), np.array([[0, 1], [(2, 3), 4]])), #this is not correctly unflattened, but can not even be created like that in standard numpy, somehow autograd numpy allows this?!
]


class FlattenTest(BaseTest):
    """Tests flatten and unflatten.
    """
    def mixed_iterable_equal(self, a, b):
        """We need a way of comparing nested mixed iterables that also
        checks that the types of sub-itreables match and that those
        of the elements compare to equal. This method does that.
        """
        print("mixed_iterable_equal(\n"+str(a)+",\n"+str(b)+")\nof types "+str(type(a))+("!=" if type(a) != type(b) else "=")+str(type(b)))
        if isinstance(a, types.GeneratorType):
            a = list(a)
        if isinstance(b, types.GeneratorType):
            b = list(b)
        if isinstance(a, collections.Iterable) or isinstance(b, collections.Iterable):
            if type(a) != type(b):
                print("mixed_iterable_equal: returning False because type(a)="+str(type(a))+"!="+str(type(b))+"=type(b)")
                return False
            if isinstance(a, np.ndarray):
                a_len = a.shape[0] if a.shape else 0
            else:
                a_len = len(a)
            if isinstance(b, np.ndarray):
                b_len = b.shape[0] if b.shape else 0
            else:
                b_len = len(b)
            print("a_len="+str(a_len)+" b_len="+str(b_len))
            if a_len != b_len:
                print("mixed_iterable_equal: returning False because a_len="+str(a_len)+"!="+str(b_len)+"=b_len")
                return False
            if a_len > 1:
                return np.all([self.mixed_iterable_equal(a[i], b[i]) for i in range(a_len)])

        print("mixed_iterable_equal: comparing result="+str(a == b))
        return a == b

    def test_depth_first_jagged_mixed(self):

        for r, a in crazy_arrays:
            self.assertTrue(self.mixed_iterable_equal(list(flatten(a)), list(r)), "flatten test failed for "+str(r)+", "+str(a))#todo: remove list() around flatten?

            a_unflattened = unflatten(r, a)
            self.assertTrue(self.mixed_iterable_equal(a_unflattened, a), "unflatten test failed for "+str(r)+", "+str(a))


    def test_flatten_list(self):
        "Tests that flatten successfully flattens multidimensional arrays."
        self.logTestName()
        flat = a
        for s in a_shapes:
            reshaped = list(np.reshape(flat, s))
            flattened = np.array([x for x in flatten(reshaped)])

            self.assertEqual(flattened.shape, flat.shape)
            self.assertAllEqual(flattened, flat)


    def test_unflatten_list(self):
        "Tests that _unflatten successfully unflattens multidimensional arrays."
        self.logTestName()
        flat = a
        for s in a_shapes:
            reshaped = list(np.reshape(flat, s))
            unflattened = np.array([x for x in unflatten(flat, reshaped)])

            self.assertEqual(unflattened.shape, np.array(reshaped).shape)
            self.assertAllEqual(unflattened, reshaped)

        with self.assertRaisesRegex(TypeError, 'Unsupported type in the model'):
            model = lambda x: x # not a valid model for unflatten
            unflatten(flat, model)

        with self.assertRaisesRegex(ValueError, 'Flattened iterable has more elements than the model'):
            unflatten(np.concatenate([flat, flat]), reshaped)


    def test_flatten_np_array(self):
        "Tests that flatten successfully flattens multidimensional arrays."
        self.logTestName()
        flat = a
        for s in a_shapes:
            reshaped = np.reshape(flat, s)
            flattened = np.array([x for x in flatten(reshaped)])

            self.assertEqual(flattened.shape, flat.shape)
            self.assertAllEqual(flattened, flat)


    def test_unflatten_np_array(self):
        "Tests that _unflatten successfully unflattens multidimensional arrays."
        self.logTestName()
        flat = a
        for s in a_shapes:
            reshaped = np.reshape(flat, s)
            unflattened = np.array([x for x in unflatten(flat, reshaped)])

            self.assertEqual(unflattened.shape, reshaped.shape)
            self.assertAllEqual(unflattened, reshaped)

        with self.assertRaisesRegex(TypeError, 'Unsupported type in the model'):
            model = lambda x: x # not a valid model for unflatten
            unflatten(flat, model)

        with self.assertRaisesRegex(ValueError, 'Flattened iterable has more elements than the model'):
            unflatten(np.concatenate([flat, flat]), reshaped)


if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', utils sub-module.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (FlattenTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
