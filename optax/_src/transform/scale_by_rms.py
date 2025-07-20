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
"""Rescale updates by the root of the exp. moving avg of the square."""

from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics

class ScaleByRmsState(NamedTuple):
    """State for exponential root mean-squared (RMS)-normalized updates."""
    nu: base.Updates

class ScaleByRmsWithCountState(NamedTuple):
    """State for exponential root mean-squared (RMS)-normalized updates."""
    count: jax.Array  # shape=(), dtype=jnp.int32.
    nu: base.Updates

def scale_by_rms(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    eps_in_sqrt: bool = True,
    bias_correction: bool = False,
) -> base.GradientTransformation:
    """Rescale updates by the root of the exp. moving avg of the square."""
    def init_fn(params):
        nu = otu.tree_full_like(params, initial_scale)
        if bias_correction:
            return ScaleByRmsWithCountState(
                count=jnp.zeros([], jnp.int32), nu=nu
            )
        else:
            return ScaleByRmsState(nu=nu)

    def update_fn(updates, state, params=None):
        del params
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, decay, 2)
        if bias_correction:
            count_inc = numerics.safe_increment(state.count)
            nu_hat = otu.tree_bias_correction(nu, decay, count_inc)
        else:
            count_inc = jnp.asarray(0)
            nu_hat = nu
        if eps_in_sqrt:
            scaling = jax.tree.map(lambda n: jax.lax.rsqrt(n + eps), nu_hat)
        else:
            scaling = jax.tree.map(lambda n: 1/(jnp.sqrt(n) + eps), nu_hat)
        updates = jax.tree.map(lambda s, g: s * g, scaling, updates)
        if bias_correction:
            new_state = ScaleByRmsWithCountState(count=count_inc, nu=nu)
        else:
            new_state = ScaleByRmsState(nu=nu)
        return updates, new_state

    return base.GradientTransformation(init_fn, update_fn)
