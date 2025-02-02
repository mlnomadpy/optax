# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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

"""Euclidean projections."""

from typing import Any

import chex

import jax
from jax import flatten_util
import jax.numpy as jnp


from optax import tree_utils as otu


def projection_non_negative(pytree: Any) -> Any:
  r"""Projection onto the non-negative orthant.

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad
    \textrm{subject to} \quad p \ge 0

  where :math:`x` is the input pytree.

  Args:
    pytree: pytree to project.
  Returns:
    projected pytree, with the same structure as ``pytree``.
  """
  return jax.tree.map(jax.nn.relu, pytree)


def _clip_safe(leaf, lower, upper):
  return jnp.clip(jnp.asarray(leaf), lower, upper)


def projection_box(pytree: Any, lower: Any, upper: Any) -> Any:
  r"""Projection onto box constraints.

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    \text{lower} \le p \le \text{upper}

  where :math:`x` is the input pytree.

  Args:
    pytree: pytree to project.
    lower:  lower bound, a scalar or pytree with the same structure as
      ``pytree``.
    upper:  upper bound, a scalar or pytree with the same structure as
      ``pytree``.
  Returns:
    projected pytree, with the same structure as ``pytree``.
  """
  return jax.tree.map(_clip_safe, pytree, lower, upper)


def projection_hypercube(pytree: Any, scale: Any = 1.0) -> Any:
  r"""Projection onto the (unit) hypercube.

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    0 \le p \le \text{scale}

  where :math:`x` is the input pytree.

  By default, we project to the unit hypercube (`scale=1.0`).

  This is a convenience wrapper around
  :func:`projection_box <optax.projections.projection_box>`.

  Args:
    pytree: pytree to project.
    scale: scale of the hypercube, a scalar or a pytree (default: 1.0).
  Returns:
    projected pytree, with the same structure as ``pytree``.
  """
  return projection_box(pytree, lower=0.0, upper=scale)


@jax.custom_jvp
def _projection_unit_simplex(values: chex.Array) -> chex.Array:
  """Projection onto the unit simplex."""
  s = 1.0
  n_features = values.shape[0]
  u = jnp.sort(values)[::-1]
  cumsum_u = jnp.cumsum(u)
  ind = jnp.arange(n_features) + 1
  cond = s / ind + (u - cumsum_u / ind) > 0
  idx = jnp.count_nonzero(cond)
  return jax.nn.relu(s / idx + (values - cumsum_u[idx - 1] / idx))


@_projection_unit_simplex.defjvp
def _projection_unit_simplex_jvp(
    primals: list[chex.Array], tangents: list[chex.Array]
) -> tuple[chex.Array, chex.Array]:
  values, = primals
  values_dot, = tangents
  primal_out = _projection_unit_simplex(values)
  supp = primal_out > 0
  card = jnp.count_nonzero(supp)
  tangent_out = supp * values_dot - (jnp.dot(supp, values_dot) / card) * supp
  return primal_out, tangent_out


def projection_simplex(pytree: Any,
                       scale: chex.Numeric = 1.0) -> Any:
  r"""Projection onto a simplex.

  This function solves the following constrained optimization problem,
  where ``x`` is the input pytree.

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    p \ge 0, p^\top 1 = \text{scale}

  By default, the projection is onto the probability simplex (unit simplex).

  Args:
    pytree: pytree to project.
    scale: value the projected pytree should sum to (default: 1.0).
  Returns:
    projected pytree, a pytree with the same structure as ``pytree``.

  .. versionadded:: 0.2.3

  Example:

    Here is an example using a pytree::

      >>> import jax.numpy as jnp
      >>> from optax import tree_utils, projections
      >>> pytree = {"w": jnp.array([2.5, 3.2]), "b": 0.5}
      >>> tree_utils.tree_sum(pytree)
      6.2
      >>> new_pytree = projections.projection_simplex(pytree)
      >>> tree_utils.tree_sum(new_pytree)
      1.0000002
  """
  if scale is None:
    scale = 1.0

  values, unravel_fn = flatten_util.ravel_pytree(pytree)
  new_values = scale * _projection_unit_simplex(values / scale)

  return unravel_fn(new_values)


def projection_l1_sphere(pytree: Any, scale: float = 1.0) -> Any:
  r"""Projection onto the l1 sphere.

  This function solves the following constrained optimization problem,
  where ``x`` is the input pytree.

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_1 = \text{scale}

  Args:
    pytree: pytree to project.
    scale: radius of the sphere.

  Returns:
    projected pytree, with the same structure as ``pytree``.
  """
  tree_abs = jax.tree.map(jnp.abs, pytree)
  tree_sign = jax.tree.map(jnp.sign, pytree)
  tree_abs_proj = projection_simplex(tree_abs, scale)
  return otu.tree_mul(tree_sign, tree_abs_proj)


def projection_l1_ball(pytree: Any, scale: float = 1.0) -> Any:
  r"""Projection onto the l1 ball.

  This function solves the following constrained optimization problem,
  where ``x`` is the input pytree.

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_1 \le \text{scale}

  Args:
    pytree: pytree to project.
    scale: radius of the ball.

  Returns:
    projected pytree, with the same structure as ``pytree``.

  .. versionadded:: 0.2.4

  Example:

      >>> import jax.numpy as jnp
      >>> from optax import tree_utils, projections
      >>> pytree = {"w": jnp.array([2.5, 3.2]), "b": 0.5}
      >>> tree_utils.tree_l1_norm(pytree)
      Array(6.2, dtype=float32)
      >>> new_pytree = projections.projection_l1_ball(pytree)
      >>> tree_utils.tree_l1_norm(new_pytree)
      Array(1.0000002, dtype=float32)
  """
  l1_norm = otu.tree_l1_norm(pytree)
  return jax.lax.cond(l1_norm <= scale,
                      lambda pytree: pytree,
                      lambda pytree: projection_l1_sphere(pytree, scale),
                      operand=pytree)


def projection_l2_sphere(pytree: Any, scale: float = 1.0) -> Any:
  r"""Projection onto the l2 sphere.

  This function solves the following constrained optimization problem,
  where ``x`` is the input pytree.

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_2 = \text{value}

  Args:
    pytree: pytree to project.
    scale: radius of the sphere.

  Returns:
    projected pytree, with the same structure as ``pytree``.

  .. versionadded:: 0.2.4
  """
  factor = scale / otu.tree_l2_norm(pytree)
  return otu.tree_scalar_mul(factor, pytree)


def projection_l2_ball(pytree: Any, scale: float = 1.0) -> Any:
  r"""Projection onto the l2 ball.

  This function solves the following constrained optimization problem,
  where ``x`` is the input pytree.

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_2 \le \text{scale}

  Args:
    pytree: pytree to project.
    scale: radius of the ball.

  Returns:
    projected pytree, with the same structure as ``pytree``.

  .. versionadded:: 0.2.4
  """
  l2_norm = otu.tree_l2_norm(pytree)
  factor = scale / l2_norm
  return jax.lax.cond(l2_norm <= scale,
                      lambda pytree: pytree,
                      lambda pytree: otu.tree_scalar_mul(factor, pytree),
                      operand=pytree)


def projection_linf_ball(pytree: Any, scale: float = 1.0) -> Any:
  r"""Projection onto the l-infinity ball.

  This function solves the following constrained optimization problem,
  where ``x`` is the input pytree.

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_{\infty} \le \text{scale}

  Args:
    pytree: pytree to project.
    scale: radius of the ball.

  Returns:
    projected pytree, with the same structure as ``pytree``.
  """
  lower_tree = otu.tree_full_like(pytree, -scale)
  upper_tree = otu.tree_full_like(pytree, scale)
  return projection_box(pytree, lower=lower_tree, upper=upper_tree)
