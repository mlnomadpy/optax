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
"""Tests for optax.losses.classification.softmax_cross_entropy."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jax.test_util as jaxtest
import numpy as np
from optax.losses.classification import softmax_cross_entropy

class SoftmaxCrossEntropyTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.ys = np.array(
            [
                [10.0, 1.0, -2.0],
                [1.0, 4.0, 0.2],
            ],
            dtype=np.float32,
        )
        self.ts = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.exp = np.array(
            [
                9.00013,
                3.0696733,
            ],
            dtype=np.float32,
        )

    @chex.all_variants
    def test_scalar(self):
        np.testing.assert_allclose(
            self.variant(softmax_cross_entropy)(self.ys[0], self.ts[0]),
            self.exp[0],
            atol=1e-4,
        )

    @chex.all_variants
    def test_batched(self):
        np.testing.assert_allclose(
            self.variant(softmax_cross_entropy)(self.ys, self.ts),
            self.exp,
            atol=1e-4,
        )

    def test_gradient(self):
        jaxtest.check_grads(
            softmax_cross_entropy,
            (self.ys[:2], self.ts[:2]),
            order=1,
        )

    @parameterized.parameters(dict(size=5), dict(size=10))
    def test_mask(self, size):
        preds = np.random.normal(size=size)
        targets = np.random.dirichlet(np.ones(size))
        mask = np.random.randint(2, size=size, dtype=bool)
        x = softmax_cross_entropy(preds[mask], targets[mask])
        y = softmax_cross_entropy(preds, targets, where=mask)
        np.testing.assert_allclose(x, y, atol=1e-4)

    @parameterized.parameters(
        dict(axis=0, shape=[4, 5, 6]),
        dict(axis=1, shape=[4, 5, 6]),
        dict(axis=2, shape=[4, 5, 6]),
    )
    def test_axis(self, shape, axis):
        preds = np.random.normal(size=shape)
        targets = np.random.dirichlet(np.ones(shape[-1]), size=shape[:-1])
        x = softmax_cross_entropy(preds, targets, axis=axis)
        y = softmax_cross_entropy(
            np.moveaxis(preds, axis, -1),
            np.moveaxis(targets, axis, -1),
        )
        np.testing.assert_allclose(x, y, atol=1e-4)

if __name__ == "__main__":
    absltest.main()
