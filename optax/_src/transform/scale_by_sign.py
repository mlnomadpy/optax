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
"""Scale updates by their sign (for signSGD and similar)."""

import jax
from optax._src import base

def scale_by_sign() -> base.GradientTransformation:
    """Scale updates by their sign (for signSGD and similar)."""
    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree.map(jax.numpy.sign, updates)
        return updates, state
    return base.GradientTransformation(base.init_empty_state, update_fn)
