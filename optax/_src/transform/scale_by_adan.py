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
"""Rescale updates according to the Adan algorithm."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics

class ScaleByAdanState(NamedTuple):
    m: base.Updates
    v: base.Updates
    n: base.Updates
    g: base.Updates
    t: jax.Array

def scale_by_adan(
    b1: float = 0.98,
    b2: float = 0.92,
    b3: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
) -> base.GradientTransformation:
    """Rescale updates according to the Adan algorithm."""
    def init_fn(params):
        return ScaleByAdanState(
            m=otu.tree_zeros_like(params),
            v=otu.tree_zeros_like(params),
            n=otu.tree_zeros_like(params),
            g=otu.tree_zeros_like(params),
            t=jnp.zeros([], jnp.int32),
        )
    def update_fn(updates, state, params=None):
        del params
        g = updates
        diff = otu.tree_where(
            state.t == 0,
            otu.tree_zeros_like(g),
            otu.tree_sub(g, state.g),
        )
        m = otu.tree_update_moment(g, state.m, b1, 1)
        v = otu.tree_update_moment(diff, state.v, b2, 1)
        sq = otu.tree_add_scalar_mul(g, 1 - b2, diff)
        n = otu.tree_update_moment_per_elem_norm(sq, state.n, b3, 2)
        t = numerics.safe_increment(state.t)
        m_hat = otu.tree_bias_correction(m, b1, t)
        v_hat = otu.tree_bias_correction(v, b2, t)
        n_hat = otu.tree_bias_correction(n, b3, t)
        u = otu.tree_add_scalar_mul(m_hat, 1 - b2, v_hat)
        denom = jax.tree.map(lambda n_hat: jnp.sqrt(n_hat + eps_root) + eps, n_hat)
        u = otu.tree_div(u, denom)
        new_state = ScaleByAdanState(
            m=m,
            v=v,
            n=n,
            g=g,
            t=t,
        )
        return u, new_state
    return base.GradientTransformation(init_fn, update_fn)
