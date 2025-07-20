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
"""Rescale updates according to the Yogi algorithm."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics

class ScaleByAdamState(NamedTuple):
    count: jax.Array
    mu: base.Updates
    nu: base.Updates

def scale_by_yogi(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3,
    eps_root: float = 0.0,
    initial_accumulator_value: float = 1e-6
) -> base.GradientTransformation:
    """Rescale updates according to the Yogi algorithm."""
    def init_fn(params):
        mu = otu.tree_full_like(params, initial_accumulator_value)
        nu = otu.tree_full_like(params, initial_accumulator_value)
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)
    def update_fn(updates, state, params=None):
        del params
        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        nu = jax.tree.map(
            lambda g, v: v - (1 - b2) * jnp.sign(v - numerics.abs_sq(g)) * numerics.abs_sq(g),
            updates, state.nu)
        count_inc = numerics.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)
    return base.GradientTransformation(init_fn, update_fn)
