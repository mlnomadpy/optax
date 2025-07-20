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
"""Tests for `centralize`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax.numpy as jnp

from optax._src.transform import centralize


class CentralizeTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('1d', [1.0, 2.0], [1.0, 2.0]),
      ('2d', [[1.0, 2.0], [3.0, 4.0]], [[-0.5, 0.5], [-0.5, 0.5]]),
      ('3d', [[[1., 2.], [3., 4.]],
              [[5., 6.], [7., 8.]]], [[[-1.5, -0.5], [0.5, 1.5]],
                                      [[-1.5, -0.5], [0.5, 1.5]]]),
  ])
  def test_centralize(self, inputs, outputs):
    inputs = jnp.asarray(inputs)
    outputs = jnp.asarray(outputs)
    centralizer = centralize.centralize()
    centralized_inputs, _ = centralizer.update(inputs, {})
    chex.assert_trees_all_close(centralized_inputs, outputs)


if __name__ == '__main__':
  absltest.main()
