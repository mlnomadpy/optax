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
"""Tests for optax.losses.classification.hinge_loss and multiclass_hinge_loss."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax.losses.classification import hinge_loss, multiclass_hinge_loss

class HingeTest(parameterized.TestCase):
    def test_binary(self):
        label = jnp.array(1)
        signed_label = jnp.array(2.0 * label - 1.0)
        score = jnp.array(10.0)
        def reference_impl(label, logit):
            return jax.nn.relu(1 - logit * (2.0 * label - 1.0))
        expected = reference_impl(label, score)
        result = hinge_loss(score, signed_label)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_batched_binary(self):
        labels = jnp.array([1, 0])
        signed_labels = jnp.array(2.0 * labels - 1.0)
        scores = jnp.array([10.0, 20.0])
        def reference_impl(label, logit):
            return jax.nn.relu(1 - logit * (2.0 * label - 1.0))
        expected = jax.vmap(reference_impl)(labels, scores)
        result = hinge_loss(scores, signed_labels)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_multi_class(self):
        label = jnp.array(1)
        scores = jnp.array([10.0, 3.0])
        def reference_impl(label, scores):
            one_hot_label = jax.nn.one_hot(label, scores.shape[-1])
            return jnp.max(scores + 1.0 - one_hot_label) - scores[label]
        expected = reference_impl(label, scores)
        result = multiclass_hinge_loss(scores, label)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_batched_multi_class(self):
        label = jnp.array([1, 0])
        scores = jnp.array([[10.0, 3.0], [11.0, -2.0]])
        def reference_impl(label, scores):
            one_hot_label = jax.nn.one_hot(label, scores.shape[-1])
            return jnp.max(scores + 1.0 - one_hot_label) - scores[label]
        expected = jax.vmap(reference_impl)(label, scores)
        result = multiclass_hinge_loss(scores, label)
        np.testing.assert_allclose(result, expected, atol=1e-4)

if __name__ == "__main__":
    absltest.main()
