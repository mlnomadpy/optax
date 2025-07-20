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
"""Scale updates by trust ratio (for LARS/LAMB)."""

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics

def scale_by_trust_ratio(
    min_norm: float = 1e-6,
    trust_coefficient: float = 0.001,
    eps: float = 0.0
) -> base.GradientTransformation:
    """Scale updates by trust ratio (for LARS/LAMB)."""
    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        def _scale_update(update, param):
            param_norm = numerics.safe_norm(param, min_norm)
            update_norm = numerics.safe_norm(update, min_norm)
            trust_ratio = trust_coefficient * param_norm / (update_norm + eps)
            return update * trust_ratio
        updates = jax.tree.map(_scale_update, updates, params)
        return updates, state
    return base.GradientTransformation(base.init_empty_state, update_fn)
