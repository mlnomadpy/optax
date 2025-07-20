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
"""Tests for optax.losses.classification.sigmoid_binary_cross_entropy."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from optax.losses.classification import sigmoid_binary_cross_entropy

class SigmoidCrossEntropyTest(parameterized.TestCase):
    @parameterized.parameters(
        dict(
            preds=np.array([-1e09, -1e-09]),
            labels=np.array([1.0, 0.0]),
            expected=5e08,
        ),
        dict(
            preds=np.array([-1e09, -1e-09]),
            labels=np.array([0.0, 1.0]),
            expected=0.3465736,
        ),
        dict(
            preds=np.array([1e09, 1e-09]),
            labels=np.array([1.0, 0.0]),
            expected=0.3465736,
        ),
        dict(
            preds=np.array([1e09, 1e-09]),
            labels=np.array([0.0, 1.0]),
            expected=5e08,
        ),
        dict(
            preds=np.array([-1e09, 1e-09]),
            labels=np.array([1.0, 0.0]),
            expected=5e08,
        ),
        dict(
            preds=np.array([-1e09, 1e-09]),
            labels=np.array([0.0, 1.0]),
            expected=0.3465736,
        ),
        dict(
            preds=np.array([1e09, -1e-09]),
            labels=np.array([1.0, 0.0]),
            expected=0.3465736,
        ),
        dict(
            preds=np.array([1e09, -1e-09]),
            labels=np.array([0.0, 1.0]),
            expected=5e08,
        ),
        dict(
            preds=np.array([0.0, 0.0]),
            labels=np.array([1.0, 0.0]),
            expected=0.6931472,
        ),
        dict(
            preds=np.array([0.0, 0.0]),
            labels=np.array([0.0, 1.0]),
            expected=0.6931472,
        ),
    )
    def testSigmoidCrossEntropy(self, preds, labels, expected):
        tested = jnp.mean(sigmoid_binary_cross_entropy(preds, labels))
        np.testing.assert_allclose(tested, expected, rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
    absltest.main()
