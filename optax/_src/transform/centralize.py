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
"""Centralize gradients."""

from optax._src import base


def _subtract_mean(g):
  if len(g.shape) > 1:
    return g - g.mean(tuple(range(1, len(g.shape))), keepdims=True)
  else:
    return g


CentralState = base.EmptyState


def centralize() -> base.GradientTransformation:
  """Centralize gradients.

  References:
    [Yong et al, 2020](https://arxiv.org/abs/2004.01461)

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return CentralState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree.map(_subtract_mean, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)
