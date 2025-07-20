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
"""Rescale updates according to the NovoGrad algorithm."""

from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics

class ScaleByNovoGradState(NamedTuple):
    count: jax.Array
    mu: base.Updates
    nu: base.Updates

def scale_by_novograd(
    b1: float = 0.95,
    b2: float = 0.98,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mu_dtype: Optional[jax.typing.DTypeLike] = None
) -> base.GradientTransformation:
    """Rescale updates according to the NovoGrad algorithm."""
    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        nu = otu.tree_zeros_like(params)
        return ScaleByNovoGradState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)
    def update_fn(updates, state, params=None):
        del params
        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        denom = jax.tree.map(lambda v: jnp.sqrt(v + eps_root) + eps, nu_hat)
        updates = jax.tree.map(lambda m, d: m / d, mu_hat, denom)
        if weight_decay != 0.0:
            updates = jax.tree.map(lambda u, p: u + weight_decay * p, updates, params)
        return updates, ScaleByNovoGradState(count=count_inc, mu=mu, nu=nu)
    return base.GradientTransformation(init_fn, update_fn)
