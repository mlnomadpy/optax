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
"""Tests of gradient transformations."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import combine
from optax._src import transform
from optax._src import update
import optax.tree_utils as otu

STEPS = 50
LR = 1e-2


class TransformTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))
    self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

  @chex.all_variants
  @parameterized.named_parameters([
      ('adadelta', transform.scale_by_adadelta),
      ('adam', transform.scale_by_adam),
      ('adamax', transform.scale_by_adamax),
      ('adan', transform.scale_by_adan),
      ('lion', transform.scale_by_lion),
      ('polyak', transform.scale_by_polyak),
      ('rmsprop', transform.scale_by_rms),
      ('stddev', transform.scale_by_stddev),
      ('trust_ratio', transform.scale_by_trust_ratio),
      ('param_block_norm', transform.scale_by_param_block_norm),
      ('param_block_rms', transform.scale_by_param_block_rms),
      ('distance_over_gradients', transform.scale_by_distance_over_gradients),
      ('normalize_by_update_norm', transform.normalize_by_update_norm),
  ])
  def test_scalers(self, scaler_constr):
    params = self.init_params

    scaler = scaler_constr()
    init_fn = self.variant(scaler.init)
    transform_fn = self.variant(scaler.update)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)

    if scaler_constr.__name__ == 'scale_by_polyak':
      extra_args = {'value': jnp.array(0.0)}
    else:
      extra_args = {}
    updates, state = transform_fn(
        self.per_step_updates, state, params, **extra_args
    )
    chex.assert_tree_all_finite((params, updates, state))
    jax.tree.map(lambda *args: chex.assert_equal_shape(args), params, updates)

  def test_scale_by_optimistic_gradient(self):
    opt = transform.scale_by_optimistic_gradient()

    state = opt.init(jnp.asarray(10.0))

    grad_0 = jnp.asarray(2.0)
    opt_grad_0, state = opt.update(grad_0, state)

    grad_1 = jnp.asarray(3.0)
    opt_grad_1, state = opt.update(grad_1, state)

    grad_2 = jnp.asarray(4.0)
    opt_grad_2, _ = opt.update(grad_2, state)

    with self.subTest('Check initial update is correct'):
      # see https://github.com/google-deepmind/optax/issues/1082
      # initial step should yield 2 * grad_0 - grad_0 = grad_0
      chex.assert_trees_all_close(opt_grad_0, grad_0)

    with self.subTest('Check second update is correct'):
      chex.assert_trees_all_close(opt_grad_1, 2 * grad_1 - grad_0)

    with self.subTest('Check third update is correct'):
      chex.assert_trees_all_close(opt_grad_2, 2 * grad_2 - grad_1)

  def test_scale_by_polyak_l1_norm(self, tol=1e-10):
    """Polyak step-size on L1 norm."""
    # for this objective, the Polyak step-size has an exact model and should
    # converge to the minimizer in one step
    objective = lambda x: jnp.abs(x).sum()

    init_params = jnp.array([1.0, -1.0])
    polyak = transform.scale_by_polyak()
    polyak_state = polyak.init(init_params)
    # check that polyak state raises an error if it called without a value
    with self.assertRaises(TypeError):
      polyak.update(self.per_step_updates, polyak_state, init_params)

    value, grad = jax.value_and_grad(objective)(init_params)
    updates, _ = polyak.update(
        grad, polyak_state, init_params, value=value
    )
    # check that objective at (init_params - updates) is smaller than tol
    print(grad, value, updates)
    self.assertLess(objective(init_params - updates), tol)

  def test_rms_match_adam(self):
    """Test scale_by_rms add_eps_in_sqrt=False matches scale_by_adam(b1=0)."""
    fun = lambda x: otu.tree_l2_norm(x, squared=True)

    rms = transform.scale_by_rms(
        decay=0.999, eps_in_sqrt=False, bias_correction=True
    )
    rms_params = self.init_params
    rms_state = rms.init(self.init_params)

    adam = transform.scale_by_adam(b1=0)
    adam_params = self.init_params
    adam_state = adam.init(self.init_params)

    for _ in range(5):
      rms_grads = jax.grad(fun)(rms_params)
      rms_updates, rms_state = rms.update(rms_grads, rms_state)
      rms_params = update.apply_updates(rms_params, rms_updates)

      adam_grads = jax.grad(fun)(adam_params)
      adam_updates, adam_state = adam.update(adam_grads, adam_state)
      adam_params = update.apply_updates(adam_params, adam_updates)

    chex.assert_trees_all_close(adam_params, rms_params)

if __name__ == '__main__':
  absltest.main()
