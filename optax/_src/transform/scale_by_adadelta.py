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
"""Rescale updates according to the Adadelta algorithm."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base

class ScaleByAdaDeltaState(NamedTuple):
    e_g: base.Updates
    e_x: base.Updates

def scale_by_adadelta(
    rho: float = 0.9, eps: float = 1e-6
) -> base.GradientTransformation:
    """Rescale updates according to the Adadelta algorithm."""
    def init_fn(params):
        e_g = otu.tree_zeros_like(params)
        e_x = otu.tree_zeros_like(params)
        return ScaleByAdaDeltaState(e_g=e_g, e_x=e_x)
    def update_fn(updates, state, params=None):
        del params
        e_g = otu.tree_update_moment(updates, state.e_g, rho, 2)
        updates = jax.tree.map(
            lambda g, cur_e_g, prev_e_x: (
                jnp.sqrt(prev_e_x + eps) / jnp.sqrt(cur_e_g + eps)
            ) * g,
            updates,
            e_g,
            state.e_x,
        )
        e_x = otu.tree_update_moment(updates, state.e_x, rho, 2)
        return updates, ScaleByAdaDeltaState(e_g=e_g, e_x=e_x)
    return base.GradientTransformation(init_fn, update_fn)
