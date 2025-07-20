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
"""Rescale updates according to the Lion algorithm."""

from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils

class ScaleByLionState(NamedTuple):
    count: jax.Array
    mu: base.Updates

def scale_by_lion(
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
) -> base.GradientTransformation:
    """Rescale updates according to the Lion algorithm."""
    mu_dtype = utils.canonicalize_dtype(mu_dtype)
    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        return ScaleByLionState(count=jnp.zeros([], jnp.int32), mu=mu)
    def update_fn(updates, state, params=None):
        del params
        updates_new = jax.tree.map(
            lambda g, m: jnp.sign((1. - b1) * g + b1 * m), updates, state.mu)
        mu = otu.tree_update_moment(updates, state.mu, b2, 1)
        mu = otu.tree_cast(mu, mu_dtype)
        count_inc = numerics.safe_increment(state.count)
        return updates_new, ScaleByLionState(count=count_inc, mu=mu)
    return base.GradientTransformation(init_fn, update_fn)
