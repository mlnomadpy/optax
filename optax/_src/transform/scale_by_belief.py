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
"""Rescale updates according to the AdaBelief algorithm."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics

class ScaleByBeliefState(NamedTuple):
    count: jax.Array
    mu: base.Updates
    nu: base.Updates

def scale_by_belief(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16
) -> base.GradientTransformation:
    """Rescale updates according to the AdaBelief algorithm."""
    def init_fn(params):
        mu = otu.tree_zeros_like(params)
        s = otu.tree_zeros_like(params)
        return ScaleByBeliefState(count=jnp.zeros([], jnp.int32), mu=mu, nu=s)
    def update_fn(updates, state, params=None):
        del params
        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        prediction_error = jax.tree.map(
            lambda g, m: g-m, updates, state.mu)
        nu = otu.tree_update_moment_per_elem_norm(prediction_error, state.nu, b2, 2)
        nu = jax.tree.map(lambda v: v + eps_root, nu)
        count_inc = numerics.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None)
        return updates, ScaleByBeliefState(count=count_inc, mu=mu, nu=nu)
    return base.GradientTransformation(init_fn, update_fn)
