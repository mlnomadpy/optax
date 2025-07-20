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
"""Unit tests for apply_every transformation."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from optax._src.transform.apply_every import apply_every

class ApplyEveryTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))
        self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

    def test_apply_every(self):
        k = 4
        zero_update = (jnp.array([0., 0.]), jnp.array([0., 0.]))
        params = self.init_params
        opt = apply_every(k=k)
        state = opt.init(params)
        for i in range(12):
            updates, state = opt.update(self.per_step_updates, state, params)
            if i % k == k-1:
                # Should emit accumulated update every k steps
                expected = tuple(u * k / k for u in self.per_step_updates)
                chex.assert_trees_all_finite(updates)
                chex.assert_trees_all_finite(state)
            else:
                chex.assert_trees_all_close(updates, zero_update, atol=0.0, rtol=0.0)

if __name__ == '__main__':
    absltest.main()
