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
"""Apply updates every k steps (for gradient accumulation)."""

from typing import NamedTuple
import jax
from optax._src import base

class ApplyEvery(NamedTuple):
    count: jax.Array
    grad_acc: base.Updates

def apply_every(
    k: int
) -> base.GradientTransformation:
    """Apply updates every k steps (for gradient accumulation)."""
    def init_fn(params):
        grad_acc = jax.tree.map(jax.numpy.zeros_like, params)
        return ApplyEvery(count=jax.numpy.zeros([], jax.numpy.int32), grad_acc=grad_acc)
    def update_fn(updates, state, params=None):
        del params
        grad_acc = jax.tree.map(lambda acc, g: acc + g, state.grad_acc, updates)
        count = state.count + 1
        apply_update = (count % k == 0)
        out = jax.tree.map(lambda acc: acc / k if apply_update else jax.numpy.zeros_like(acc), grad_acc)
        grad_acc = jax.tree.map(lambda acc: jax.numpy.zeros_like(acc) if apply_update else acc, grad_acc)
        return out, ApplyEvery(count=count, grad_acc=grad_acc)
    return base.GradientTransformation(init_fn, update_fn)
