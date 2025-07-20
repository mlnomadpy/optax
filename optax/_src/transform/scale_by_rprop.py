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
"""Rescale updates according to the Rprop algorithm."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from optax._src import base

class ScaleByRpropState(NamedTuple):
    step_sizes: base.Updates
    prev_updates: base.Updates

def scale_by_rprop(
    learning_rate: float,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    min_step_size: float = 1e-6,
    max_step_size: float = 50.0,
) -> base.GradientTransformation:
    """Rescale updates according to the Rprop algorithm."""
    def init_fn(params):
        step_sizes = jax.tree.map(lambda _: learning_rate, params)
        prev_updates = jax.tree.map(jnp.zeros_like, params)
        return ScaleByRpropState(step_sizes=step_sizes, prev_updates=prev_updates)
    def update_fn(updates, state, params=None):
        del params
        def update_step_size(step_size, prev_update, update):
            sign = jnp.sign(update * prev_update)
            step_size = jnp.where(sign > 0, jnp.minimum(step_size * eta_plus, max_step_size), step_size)
            step_size = jnp.where(sign < 0, jnp.maximum(step_size * eta_minus, min_step_size), step_size)
            return step_size
        step_sizes = jax.tree.map(update_step_size, state.step_sizes, state.prev_updates, updates)
        new_updates = jax.tree.map(lambda u, s: jnp.sign(u) * s, updates, step_sizes)
        return new_updates, ScaleByRpropState(step_sizes=step_sizes, prev_updates=updates)
    return base.GradientTransformation(init_fn, update_fn)
