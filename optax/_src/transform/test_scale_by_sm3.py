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
"""Tests for `scale_by_sm3`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp

from optax._src import numerics
from optax._src.transform import scale_by_sm3


class SM3Test(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (
        jnp.array([[1., 2.], [3., 4.]]),
        jnp.array([1., 2., 3., 4.]),
    )
    self.per_step_updates = (
        jnp.array([[500., 5.], [300., 3.]]),
        jnp.array([500., 5., 300., 3.]),
    )

  @chex.all_variants
  def test_scale_by_sm3(self, variant):
    params = self.init_params
    scaler = scale_by_sm3.scale_by_sm3()
    init_fn = variant(scaler.init)
    transform_fn = variant(scaler.update)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)

    updates, state = transform_fn(self.per_step_updates, state, params)
    chex.assert_tree_all_finite((params, updates, state))
    jax.tree.map(
        lambda *args: chex.assert_equal_shape(args), params, updates)

  def test_complex_numbers(self):
    """Tests that SM3 rejects complex parameters."""
    scaler = scale_by_sm3.scale_by_sm3()
    params = (jnp.array([1j, 2j]),)
    with self.assertRaises(ValueError):
      scaler.init(params)

  def test_numerics(self):
    """Tests that SM3 is numerically stable."""
    for b1 in [0.9, 1.0]:
      for b2 in [0.9, 1.0]:
        scaler = scale_by_sm3.scale_by_sm3(b1=b1, b2=b2)
        params = (jnp.array([1.0, 2.0]),)
        state = scaler.init(params)
        updates = (jnp.array([1e-10, 1e-10]),)
        updates, _ = scaler.update(updates, state, params)
        chex.assert_tree_all_finite(updates)


if __name__ == '__main__':
  absltest.main()
