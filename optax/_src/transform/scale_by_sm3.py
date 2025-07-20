# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Scale updates by sm3."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import numerics
from optax._src import utils as otu


class ScaleBySM3State(NamedTuple):
  """State for the SM3 algorithm."""
  mu: base.Updates
  nu: base.Updates


def _reject_complex(params):
  """Reject complex parameters."""
  otu.tree_map_with_path(
      lambda path, arr: arr.dtype not in [jnp.complex64, jnp.complex128]
      or ValueError(
          f'SM3 does not support complex parameters, but got {path} with'
          f' dtype {arr.dtype}'
      ),
      params,
  )


def scale_by_sm3(
    b1: float = 0.9,
    b2: float = 1.0,
    eps: float = 1e-8
) -> base.GradientTransformation:
  """Scale updates by `sm3`.

  References:
    [Anil et. al 2019](https://arxiv.org/abs/1901.11150)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.

  Returns:
    A `GradientTransformation` object.
  """

  def zeros_for_dim(p):
    return [jnp.zeros([s], dtype=p.dtype) for s in p.shape]

  def init_fn(params):
    _reject_complex(params)
    mu = jax.tree.map(zeros_for_dim, params)
    nu = otu.tree_zeros_like(params)
    return ScaleBySM3State(mu, nu)

  def _expanded_shape(shape, axis):
    # Replaces a `shape` of [M, N, K] with 1 in all dimensions except for i.
    # For eg: i = 1 returns [1, N, 1].
    rank = len(shape)
    return [1] * axis + [shape[axis]] + [1] * (rank - axis - 1)

  def _new_mu(mu, g, b1):
    return [b1 * m + (1 - b1) * g for m in mu]

  def _new_nu(nu, g, b2):
    return b2 * nu + (1 - b2) * g**2

  def update_fn(updates, state, params=None):
    del params
    _reject_complex(updates)
    mu = jax.tree.map(_new_mu, state.mu, updates, b1)
    nu = jax.tree.map(_new_nu, state.nu, updates, b2)

    def _get_update(m, v):
      m_hat = [jnp.maximum(v, m_i**2) for m_i in m]
      m_hat = [
          numerics.safe_norm(jnp.reshape(m_i, [-1]))
          for m_i in m_hat
      ]
      m_hat = [
          jnp.reshape(m_i, _expanded_shape(v.shape, i))
          for i, m_i in enumerate(m_hat)
      ]
      m_hat = sum(m_hat)
      return m / (jnp.sqrt(m_hat) + eps)

    updates = jax.tree.map(_get_update, mu, nu)
    return updates, ScaleBySM3State(mu, nu)

  return base.GradientTransformation(init_fn, update_fn)
