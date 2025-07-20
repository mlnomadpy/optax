# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for optax.losses.classification.softmax_cross_entropy_with_integer_labels."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jax.test_util as jaxtest
import numpy as np
from optax.losses.classification import softmax_cross_entropy_with_integer_labels, softmax_cross_entropy

class SoftmaxCrossEntropyWithIntegerLabelsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.ys = np.array([[10.0, 1.0, -2.0], [1.0, 4.0, 0.2]], dtype=np.float32)
        self.ts = np.array([1, 0], dtype=np.int32)

    @chex.all_variants
    def test_consistent_with_softmax_cross_entropy_scalar(self):
        exp = softmax_cross_entropy(self.ys[0], jax.nn.one_hot(self.ts[0], 3))
        np.testing.assert_allclose(
            self.variant(softmax_cross_entropy_with_integer_labels)(self.ys[0], self.ts[0]),
            exp,
            rtol=1e-6,
        )

    @chex.all_variants
    def test_consistent_with_softmax_cross_entropy_batched(self):
        exp = softmax_cross_entropy(self.ys, jax.nn.one_hot(self.ts, 3))
        np.testing.assert_allclose(
            self.variant(softmax_cross_entropy_with_integer_labels)(self.ys, self.ts),
            exp,
            rtol=1e-6,
        )

    def test_gradient(self):
        jaxtest.check_grads(
            functools.partial(
                softmax_cross_entropy_with_integer_labels,
                labels=self.ts,
            ),
            (self.ys,),
            order=1,
        )

    @parameterized.parameters(
        dict(axis=0, shape=[4, 5, 6]),
        dict(axis=1, shape=[4, 5, 6]),
        dict(axis=2, shape=[4, 5, 6]),
    )
    def test_axis(self, shape, axis):
        preds = np.random.normal(size=shape)
        targets = np.random.randint(
            shape[axis], size=shape[:axis] + shape[axis + 1 :]
        )
        f = softmax_cross_entropy_with_integer_labels
        x = f(preds, targets, axis=axis)
        y = f(
            np.moveaxis(preds, axis, -1),
            targets,
        )
        np.testing.assert_allclose(x, y, atol=1e-4)

if __name__ == "__main__":
    absltest.main()
