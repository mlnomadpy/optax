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
"""Unit tests for scale_by_polyak transformation."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from optax._src.transform import scale_by_polyak

class ScaleByPolyakTest(chex.TestCase):
    def test_scale_by_polyak_l1_norm(self, tol=1e-10):
        objective = lambda x: jnp.abs(x).sum()
        init_params = jnp.array([1.0, -1.0])
        polyak = scale_by_polyak()
        polyak_state = polyak.init(init_params)
        with self.assertRaises(TypeError):
            polyak.update(jnp.array([1.0, 1.0]), polyak_state, init_params)
        value, grad = jax.value_and_grad(objective)(init_params)
        updates, _ = polyak.update(grad, polyak_state, init_params, value=value)
        self.assertLess(objective(init_params - updates), tol)

if __name__ == '__main__':
    absltest.main()
