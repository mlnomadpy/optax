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
"""Tests for optax.losses.regression.l2_loss."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
import numpy as np
from optax.losses.regression import l2_loss

class L2LossTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.ys = jnp.array([-2., -1., 0.5, 1.])
        self.ts = jnp.array([-1.5, 0., -1, 1.])
        self.exp = 0.5 * (self.ts - self.ys) ** 2

    @chex.all_variants
    def test_scalar(self):
        np.testing.assert_allclose(
            self.variant(l2_loss)(self.ys[0], self.ts[0]), self.exp[0])

    @chex.all_variants
    def test_batched(self):
        np.testing.assert_allclose(
            self.variant(l2_loss)(self.ys, self.ts), self.exp)

    @chex.all_variants
    def test_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            _ = self.variant(l2_loss)(
                self.ys, jnp.expand_dims(self.ts, axis=-1))

if __name__ == "__main__":
    absltest.main()
