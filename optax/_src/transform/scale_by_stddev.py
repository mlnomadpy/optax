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
"""Rescale updates by the root of the centered exp. moving average of squares."""

from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics

abs_sq = numerics.abs_sq

class ScaleByRStdDevState(NamedTuple):
    mu: base.Updates
    nu: base.Updates

class ScaleByRStdDevWithCountState(NamedTuple):
    count: jax.Array
    mu: base.Updates
    nu: base.Updates

def scale_by_stddev(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.,
    eps_in_sqrt: bool = True,
    bias_correction: bool = False,
) -> base.GradientTransformation:
    """Rescale updates by the root of the centered exp. moving average of squares."""
    def init_fn(params):
        mu = otu.tree_zeros_like(params)
        nu = otu.tree_full_like(params, initial_scale)
        if bias_correction:
            return ScaleByRStdDevWithCountState(
                count=jnp.zeros([], jnp.int32), mu=mu, nu=nu
            )
        else:
            return ScaleByRStdDevState(mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = otu.tree_update_moment(updates, state.mu, decay, 1)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, decay, 2)
        if bias_correction:
            count_inc = numerics.safe_increment(state.count)
            mu_hat = otu.tree_bias_correction(mu, decay, count_inc)
            nu_hat = otu.tree_bias_correction(nu, decay, count_inc)
        else:
            count_inc = jnp.asarray(0)
            mu_hat = mu
            nu_hat = nu
        if eps_in_sqrt:
            scaling = jax.tree.map(
                lambda m, n: jax.lax.rsqrt(n - abs_sq(m) + eps),
                mu_hat,
                nu_hat,
            )
        else:
            scaling = jax.tree.map(
                lambda m, n: 1/(jnp.sqrt(n - abs_sq(m)) + eps),
                mu_hat,
                nu_hat,
            )
        updates = jax.tree.map(
            lambda s, g: s * g, scaling, updates
        )
        if bias_correction:
            new_state = ScaleByRStdDevWithCountState(
                count=count_inc, mu=mu, nu=nu
            )
        else:
            new_state = ScaleByRStdDevState(mu=mu, nu=nu)
        return updates, new_state

    return base.GradientTransformation(init_fn, update_fn)
