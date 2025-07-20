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
"""Tests for optax.losses.regression.log_cosh."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
import numpy as np
from optax.losses.regression import log_cosh

class LogCoshTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.ys = jnp.array([500, -2., -1., 0.5, 1.])
        self.ts = jnp.array([-200, -1.5, 0., -1, 1.])
        self.exp = jnp.array([699.3068, 0.12011445, 0.4337809, 0.85544014, 0.])
        self.exp_ys_only = jnp.array([
            499.30685, 1.3250027, 0.4337809, 0.12011451, 0.43378082])

    @chex.all_variants
    def test_scalar(self):
        out = self.variant(log_cosh)(self.ys[0], self.ts[0])
        np.testing.assert_allclose(out, self.exp[0], atol=1e-5)

    @chex.all_variants
    def test_batched(self):
        out = self.variant(log_cosh)(self.ys, self.ts)
        np.testing.assert_allclose(out, self.exp, atol=1e-5)

    @chex.all_variants
    def test_scalar_predictions_only(self):
        out = self.variant(log_cosh)(self.ys[0])
        np.testing.assert_allclose(out, self.exp_ys_only[0], atol=1e-5)

    @chex.all_variants
    def test_batched_predictions_only(self):
        out = self.variant(log_cosh)(self.ys)
        np.testing.assert_allclose(out, self.exp_ys_only, atol=1e-5)

if __name__ == "__main__":
    absltest.main()
