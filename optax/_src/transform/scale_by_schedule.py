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
"""Scale updates by a learning rate schedule."""

from typing import NamedTuple, Callable
import jax
from optax._src import base

class ScaleByScheduleState(NamedTuple):
    count: jax.Array

def scale_by_learning_rate(
    learning_rate: float
) -> base.GradientTransformation:
    """Scale updates by a fixed learning rate (for legacy compatibility)."""
    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree.map(lambda g: learning_rate * g, updates)
        return updates, state
    return base.GradientTransformation(base.init_empty_state, update_fn)

def scale_by_schedule(
    schedule_fn: Callable[[int], float]
) -> base.GradientTransformation:
    """Scale updates by a learning rate schedule."""
    def init_fn(params):
        del params
        return ScaleByScheduleState(count=jax.numpy.zeros([], jax.numpy.int32))
    def update_fn(updates, state, params=None):
        del params
        lr = schedule_fn(state.count)
        updates = jax.tree.map(lambda g: lr * g, updates)
        return updates, ScaleByScheduleState(count=state.count + 1)
    return base.GradientTransformation(init_fn, update_fn)
