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
"""Tests for optax.losses.classification.safe_softmax_cross_entropy."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jax.test_util as jaxtest
import numpy as np
from optax.losses.classification import safe_softmax_cross_entropy, softmax_cross_entropy

class SafeSoftmaxCrossEntropyTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.ys = np.array(
            [
                [10.0, 1.0, -2.0],
                [1.0, 4.0, 0.2],
                [-np.inf, 0.0, 0.0],
                [-np.inf, 0.0, 0.0],
                [-np.inf, 0.0, -np.inf],
            ],
            dtype=np.float32,
        )
        self.ts = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.4, 0.3, 0.3],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.exp = np.array(
            [
                9.00013,
                3.0696733,
                0.693147,
                np.inf,
                0.0,
            ],
            dtype=np.float32,
        )

    @chex.all_variants
    def test_scalar(self):
        np.testing.assert_allclose(
            self.variant(safe_softmax_cross_entropy)(self.ys[0], self.ts[0]),
            self.exp[0],
            atol=1e-4,
        )

    @chex.all_variants
    def test_batched(self):
        np.testing.assert_allclose(
            self.variant(safe_softmax_cross_entropy)(self.ys, self.ts),
            self.exp,
            atol=1e-4,
        )

    def test_gradient(self):
        jaxtest.check_grads(
            safe_softmax_cross_entropy,
            (self.ys[:2], self.ts[:2]),
            order=1,
        )

    def test_against_plain_implementation(self):
        plain_val_and_grad = jax.value_and_grad(softmax_cross_entropy)(self.ys[0], self.ts[0])
        val_and_grad = jax.value_and_grad(safe_softmax_cross_entropy)(self.ys[0], self.ts[0])
        chex.assert_trees_all_close(plain_val_and_grad, val_and_grad, atol=1e-4)

if __name__ == "__main__":
    absltest.main()
