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
"""Unit tests for scale_by_optimistic_gradient transformation."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from optax._src.transform import scale_by_optimistic_gradient

class ScaleByOptimisticGradientTest(chex.TestCase):
    def test_scale_by_optimistic_gradient(self):
        opt = scale_by_optimistic_gradient()
        state = opt.init(jnp.asarray(10.0))
        grad_0 = jnp.asarray(2.0)
        opt_grad_0, state = opt.update(grad_0, state)
        grad_1 = jnp.asarray(3.0)
        opt_grad_1, state = opt.update(grad_1, state)
        grad_2 = jnp.asarray(4.0)
        opt_grad_2, _ = opt.update(grad_2, state)
        with self.subTest('Check initial update is correct'):
            chex.assert_trees_all_close(opt_grad_0, grad_0)
        with self.subTest('Check second update is correct'):
            chex.assert_trees_all_close(opt_grad_1, 2 * grad_1 - grad_0)
        with self.subTest('Check third update is correct'):
            chex.assert_trees_all_close(opt_grad_2, 2 * grad_2 - grad_1)

if __name__ == '__main__':
    absltest.main()
