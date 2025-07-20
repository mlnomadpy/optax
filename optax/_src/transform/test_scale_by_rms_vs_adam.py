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
"""Unit test to check scale_by_rms matches scale_by_adam(b1=0)."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from optax._src.transform import scale_by_rms, scale_by_adam
import optax.tree_utils as otu

class ScaleByRmsVsAdamTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))

    def test_rms_match_adam(self):
        fun = lambda x: otu.tree_l2_norm(x, squared=True)
        rms = scale_by_rms(decay=0.999, eps_in_sqrt=False, bias_correction=True)
        rms_params = self.init_params
        rms_state = rms.init(self.init_params)
        adam = scale_by_adam(b1=0)
        adam_params = self.init_params
        adam_state = adam.init(self.init_params)
        for _ in range(5):
            rms_grads = jax.grad(fun)(rms_params)
            rms_updates, rms_state = rms.update(rms_grads, rms_state)
            rms_params = tuple(p + u for p, u in zip(rms_params, rms_updates))
            adam_grads = jax.grad(fun)(adam_params)
            adam_updates, adam_state = adam.update(adam_grads, adam_state)
            adam_params = tuple(p + u for p, u in zip(adam_params, adam_updates))
        chex.assert_trees_all_close(adam_params, rms_params)

if __name__ == '__main__':
    absltest.main()
