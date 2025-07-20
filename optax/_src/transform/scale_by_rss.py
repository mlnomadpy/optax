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
"""Rescale updates by the root of the sum of all squared gradients to date."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics

abs_sq = numerics.abs_sq

class ScaleByRssState(NamedTuple):
    """State holding the sum of gradient squares to date."""
    sum_of_squares: base.Updates

def scale_by_rss(
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7
) -> base.GradientTransformation:
    """Rescale updates by the root of the sum of all squared gradients to date."""
    def init_fn(params):
        return ScaleByRssState(
            sum_of_squares=otu.tree_full_like(params, initial_accumulator_value))

    def update_fn(updates, state, params=None):
        del params
        sum_of_squares = jax.tree.map(
            lambda g, t: abs_sq(g) + t, updates, state.sum_of_squares)
        inv_sqrt_g_square = jax.tree.map(
            lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), sum_of_squares)
        updates = otu.tree_mul(inv_sqrt_g_square, updates)
        return updates, ScaleByRssState(sum_of_squares=sum_of_squares)

    return base.GradientTransformation(init_fn, update_fn)
