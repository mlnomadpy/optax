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
"""Unit tests for scale transformation."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from optax._src.transform.scale import scale

class ScaleTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

    def test_scale(self):
        updates = self.per_step_updates
        for i in range(1, 6):
            factor = 0.1 ** i
            rescaler = scale(factor)
            scaled_updates, _ = rescaler.update(updates, {})
            manual_updates = jax.tree.map(lambda t: t * factor, updates)
            chex.assert_trees_all_close(scaled_updates, manual_updates)

if __name__ == '__main__':
    absltest.main()
