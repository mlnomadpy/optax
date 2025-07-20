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
"""Scale updates for each param block by the norm of that block's parameters."""

import jax
from optax._src import base
from optax._src import numerics


def scale_by_param_block_norm(
    min_scale: float = 1e-3
) -> base.GradientTransformation:
    """Scale updates for each param block by the norm of that block's parameters."""
    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        updates = jax.tree.map(
            lambda u, p: u * numerics.safe_norm(p, min_scale),
            updates, params)
        return updates, state
    return base.GradientTransformation(base.init_empty_state, update_fn)
