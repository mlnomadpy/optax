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
"""Tests for optax.losses.regression.huber_loss."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np
from optax.losses.regression import huber_loss

class HuberLossTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.ys = np.array([-2.0, 0.5, 0., 0.5, 2.0, 4.0, 132.])
        self.ts = np.array([0.0, -0.5, 0., 1., 1.0, 2.0, 0.3])
        self.exp = np.array([1.5, 0.5, 0., 0.125, 0.5, 1.5, 131.2])

    @chex.all_variants
    def test_scalar(self):
        np.testing.assert_allclose(
            self.variant(huber_loss)(
                self.ys[0], self.ts[0], delta=1.0),
            self.exp[0])

    @chex.all_variants
    def test_batched(self):
        np.testing.assert_allclose(
            self.variant(huber_loss)(
                self.ys, self.ts, delta=1.0),
            self.exp)

if __name__ == "__main__":
    absltest.main()
